"""Vision-language model, LLaVA style: a ViT turns an image into patch tokens, a small MLP projects them into
the language model's embedding space, and they replace the <|image|> placeholders in the prompt:

    <|user|> <|image|> x 64  How many red shapes are there? <|assistant|> 2 <|endoftext|>

    scripts/pgx.sh python -m scratchlm.vision --name vlm --init runs/rsi/r10/ckpt.pt
"""
import argparse
import dataclasses
import json
import os
import struct
import time
import zlib
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .model import GPT, Block, Config, Linear, RMSNorm, load_checkpoint, rope_tables, silu
from .optim import AdamW, clip_grad_norm, cosine_lr
from .tasks import ASSISTANT, EOT, IMAGE, IMG, USER, chat_tokenizer, encode_chat, question, render, scene
from .tokenizer import Tokenizer

PATCH = 8
N_PATCHES = (IMG // PATCH) ** 2  # 64 image tokens


class ViT(nn.Module):
    """Image -> one token per 8x8 patch. Patch pixels are flattened and linearly embedded, then bidirectional
    Transformer blocks (the same Block as the LM, with causal=False) mix information across patches.
    Position: 1-D RoPE over the raster order of patches."""

    def __init__(self, c):
        super().__init__()
        self.c = c
        self.patch_embed = Linear(PATCH * PATCH * 3, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.norm = RMSNorm(c.d_model)
        cos, sin = rope_tables(c.d_model // c.n_head, N_PATCHES, c.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, img):  # (B, H, W, 3) floats
        B, H, W, C = img.shape
        x = img.reshape(B, H // PATCH, PATCH, W // PATCH, PATCH, C).permute(0, 1, 3, 2, 4, 5)
        x = self.patch_embed(x.reshape(B, N_PATCHES, PATCH * PATCH * C))  # (B, 64, d_vit)
        for b in self.blocks:
            x = b(x, self.cos, self.sin)
        return self.norm(x)


class VLM(nn.Module):
    def __init__(self, lm, vit, image_id):
        super().__init__()
        self.lm, self.vit, self.image_id = lm, vit, image_id
        self.proj1, self.proj2 = Linear(vit.c.d_model, lm.c.d_model), Linear(lm.c.d_model, lm.c.d_model)

    def embed(self, ids, images):
        """Token embeddings with each row's <|image|> positions overwritten, in order, by its projected patches."""
        x = self.lm.embed(ids)
        f = self.proj2(silu(self.proj1(self.vit(images))))  # (B, 64, d_model)
        return x.masked_scatter((ids == self.image_id)[..., None], f.to(x.dtype))

    def forward(self, ids, images, targets=None):
        return self.lm(ids, targets, embeds=self.embed(ids, images))

    def generate(self, ids, images, max_new):
        return self.lm.generate(ids, max_new, temperature=0, embeds=self.embed(ids, images))


def to_tensor(imgs, dev):
    return torch.from_numpy(np.stack(imgs)).to(dev).float() / 255 - 0.5


def make_batch(rng, tok, B, counts, L, dev):
    """B random (scene, question, answer) examples padded to L+1 tokens; targets -1 outside the answer."""
    imgs, rows, masks = [], [], []
    eot = tok.specials[EOT]
    for _ in range(B):
        objs = scene(rng, int(rng.integers(counts[0], counts[1] + 1)))
        q, a = question(rng, objs)
        ids, mask = encode_chat(tok, IMAGE * N_PATCHES + q, a)
        assert len(ids) <= L + 1, "raise L"
        rows.append(ids + [eot] * (L + 1 - len(ids)))
        masks.append(mask + [0] * (L + 1 - len(mask)))
        imgs.append(render(objs, rng))
    w, m = torch.tensor(rows, device=dev), torch.tensor(masks, device=dev)
    return w[:, :-1], to_tensor(imgs, dev), torch.where(m[:, 1:] == 1, w[:, 1:], -1)


def answer(vlm, tok, imgs, questions, dev):
    """Greedy replies, batched by prompt token length (the KV-cache generate needs equal-length rows)."""
    prompts = [tok.encode(USER + IMAGE * N_PATCHES + q + ASSISTANT) for q in questions]
    out, eot = [None] * len(prompts), tok.specials[EOT]
    for n in sorted({len(p) for p in prompts}):
        idx = [i for i, p in enumerate(prompts) if len(p) == n]
        g = vlm.generate(torch.tensor([prompts[i] for i in idx], device=dev), to_tensor([imgs[i] for i in idx], dev), 3)
        for i, row in zip(idx, g[:, n:].tolist()):
            out[i] = tok.decode(row[: row.index(eot)] if eot in row else row)
    return out


@torch.no_grad()
def evaluate(vlm, tok, counts, n=1000, seed=123, dev="cuda"):
    """Accuracy per question type, and accuracy over all question types per number of objects in the scene."""
    rng = np.random.default_rng(seed)
    data = []
    for i in range(n):
        objs = scene(rng, counts[i % len(counts)])
        data.append((objs, *question(rng, objs), render(objs, rng)))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        replies = answer(vlm, tok, [d[3] for d in data], [d[1] for d in data], dev)
    by_kind, by_count = {}, {}
    for (objs, q, a, _), r in zip(data, replies):
        kind = ("count_all" if q == "How many shapes are there?" else "count_color" if q.endswith("shapes are there?")
                else "count_shape" if q.startswith("How many") else "exists" if q.startswith("Is") else "color_of")
        by_kind.setdefault(kind, []).append(r == a)
        by_count.setdefault(len(objs), []).append(r == a)
    mean = lambda d: {k: round(float(np.mean(v)), 3) for k, v in sorted(d.items())}
    return {"by_kind": mean(by_kind), "by_objects": mean(by_count), "overall": round(float(np.mean([r == d[2] for d, r in zip(data, replies)])), 3)}


def save_png(path, img):
    """Minimal PNG writer (stdlib zlib), so sample scenes can go into the wiki without an image library."""
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


@dataclass
class VisionConfig:
    name: str = "vlm"
    init: str = "runs/rsi/r10/ckpt.pt"  # language model to attach the eye to
    tok: str = "data/tok_chat.json"
    freeze_lm: bool = True  # stage A: only ViT + projector learn (alignment); the LM stays as it was
    steps: int = 4000
    batch: int = 128
    lr: float = 1e-3
    warmup: int = 200
    min_count: int = 1
    max_count: int = 6  # objects per training scene; more objects are held out
    L: int = 96  # tokens per example (64 image + question + answer, padded)
    vit_layers: int = 4
    vit_dim: int = 256
    seed: int = 0


def build(vc, dev):
    tok = chat_tokenizer(Tokenizer.load(vc.tok))  # adds <|image|> if the file predates it
    lm = load_checkpoint(vc.init, dev, tok.vocab_size)
    vit = ViT(Config(vocab_size=1, ctx_len=N_PATCHES, n_layer=vc.vit_layers, n_head=vc.vit_dim // 64, d_model=vc.vit_dim, causal=False))
    return tok, VLM(lm, vit, tok.specials[IMAGE]).to(dev)


def main(vc):
    dev = "cuda"
    run = os.path.join("runs", vc.name)
    os.makedirs(run, exist_ok=True)
    torch.manual_seed(vc.seed)
    tok, vlm = build(vc, dev)
    if vc.freeze_lm:
        vlm.lm.requires_grad_(False)
    params = [p for p in vlm.parameters() if p.requires_grad]
    opt = AdamW(params)
    log = open(os.path.join(run, "log.jsonl"), "a")

    def emit(**row):
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(row, flush=True)

    emit(event="start", trainable=sum(p.numel() for p in params), vit=sum(p.numel() for p in vlm.vit.parameters()), config=dataclasses.asdict(vc))
    rng = np.random.default_rng(vc.seed)
    train_counts, all_counts = (vc.min_count, vc.max_count), list(range(1, 11))
    t0 = time.time()
    for step in range(vc.steps):
        x, imgs, y = make_batch(rng, tok, vc.batch, train_counts, vc.L, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = vlm(x, imgs, y)
        loss.backward()
        clip_grad_norm(params, 1.0)
        opt.step(cosine_lr(step, vc.lr, vc.warmup, vc.steps))
        opt.zero_grad()
        if (step + 1) % 100 == 0:
            emit(step=step + 1, loss=round(loss.item(), 4), secs=round(time.time() - t0))
        if (step + 1) % 1000 == 0 or step + 1 == vc.steps:
            vlm.eval()
            emit(step=step + 1, eval=evaluate(vlm, tok, all_counts))
            vlm.train()
    torch.save({"vlm": vlm.state_dict(), "config": dataclasses.asdict(vc)}, os.path.join(run, "ckpt.pt"))
    emit(event="done", secs=round(time.time() - t0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the VLM on synthetic scenes")
    for f in dataclasses.fields(VisionConfig):
        kind = (lambda s: s.lower() in ("1", "true", "yes")) if isinstance(f.default, bool) else type(f.default)
        ap.add_argument(f"--{f.name}", type=kind, default=f.default)
    main(VisionConfig(**vars(ap.parse_args())))

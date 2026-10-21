"""Vision-language model, LLaVA style: a ViT turns an image into patch tokens, a small MLP projects them into
the language model's embedding space, and they replace the <|image|> placeholders in the prompt:

    <|user|> <|image|> x 64  How many red shapes are there? <|assistant|> 2 <|endoftext|>

    DETACH=vlm scripts/pgx.sh python -m scratchlm.vision --name vlm                 # stage A: align the eye
    DETACH=vrsi scripts/pgx.sh python -m scratchlm.vision rsi --name vrsi          # image RSI: count past 6
"""
import argparse
import collections
import dataclasses
import json
import os
import struct
import sys
import time
import zlib
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .data import batches
from .model import GPT, Block, Config, Linear, RMSNorm, load_checkpoint, rope_tables, silu
from .optim import AdamW, clip_grad_norm, cosine_lr
from .sft import skills
from .tasks import ASSISTANT, EOT, GRID, IMAGE, IMG, USER, chat_tokenizer, encode_chat, parse_chain, question, render, row_counts, scene, sum_question
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

    def generate(self, ids, images, max_new, temperature=0.0):
        return self.lm.generate(ids, max_new, temperature=temperature, embeds=self.embed(ids, images))


def to_tensor(imgs, dev):
    return torch.from_numpy(np.stack(imgs)).to(dev).float() / 255 - 0.5


def examples(rng, n, counts, rows=False, drill=0.0, count_only=False):
    """n random (objects, image, question, answer) with a number of objects drawn from `counts`.
    A `drill` share are text-only sums on a blank image (no objects); count_only asks only "How many shapes"."""
    out = []
    for _ in range(n):
        if rng.random() < drill:
            out.append(([], render([], rng), *sum_question(rng)))
            continue
        objs = scene(rng, int(rng.choice(counts)))
        q, a = question(rng, objs, rows)
        while count_only and kind(q) != "count_all":
            q, a = question(rng, objs, rows)
        out.append((objs, render(objs, rng), q, a))
    return out


def make_batch(tok, exs, L, dev):
    """Examples padded to L+1 tokens; targets -1 outside the answer."""
    rows, masks = [], []
    eot = tok.specials[EOT]
    for _, _, q, a in exs:
        ids, mask = encode_chat(tok, IMAGE * N_PATCHES + q, a)
        assert len(ids) <= L + 1, "raise L"
        rows.append(ids + [eot] * (L + 1 - len(ids)))
        masks.append(mask + [0] * (L + 1 - len(mask)))
    w, m = torch.tensor(rows, device=dev), torch.tensor(masks, device=dev)
    return w[:, :-1], to_tensor([e[1] for e in exs], dev), torch.where(m[:, 1:] == 1, w[:, 1:], -1)


def fit(vlm, tok, next_examples, steps, lr, warmup, L, dev, every=None, text=None):
    """Train the parameters that require grad (fresh AdamW, warmup + cosine). Calls every(step) each 100 steps.
    `text`: optional endless (x, y) text batches whose LM loss is added each step (replay against forgetting)."""
    params = [p for p in vlm.parameters() if p.requires_grad]
    opt = AdamW(params)
    vlm.train()
    for step in range(steps):
        x, imgs, y = make_batch(tok, next_examples(), L, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = vlm(x, imgs, y)
            if text:
                loss = loss + vlm.lm(*next(text))[1]
        loss.backward()
        clip_grad_norm(params, 1.0)
        opt.step(cosine_lr(step, lr, warmup, steps))
        opt.zero_grad()
        if every and (step + 1) % 100 == 0:
            every(step + 1, loss.item())
    vlm.eval()


def answer(vlm, tok, imgs, questions, dev, temperature=0.0, max_new=3, chunk=2048):
    """Replies (greedy by default), batched by prompt token length (the KV-cache generate needs equal-length rows),
    at most `chunk` rows per batch."""
    prompts = [tok.encode(USER + IMAGE * N_PATCHES + q + ASSISTANT) for q in questions]
    out, eot = [None] * len(prompts), tok.specials[EOT]
    for n in sorted({len(p) for p in prompts}):
        same = [i for i, p in enumerate(prompts) if len(p) == n]
        for idx in (same[j : j + chunk] for j in range(0, len(same), chunk)):
            g = vlm.generate(torch.tensor([prompts[i] for i in idx], device=dev), to_tensor([imgs[i] for i in idx], dev), max_new, temperature)
            for i, row in zip(idx, g[:, n:].tolist()):
                out[i] = tok.decode(row[: row.index(eot)] if eot in row else row)
    return out


def kind(q):
    return ("count_all" if q == "How many shapes are there?" else "sum" if q.startswith("What is") else "count_color" if q.endswith("shapes are there?")
            else "count_shape" if q.startswith("How many") else "exists" if q.startswith("Is") else "color_of")


@torch.no_grad()
def evaluate(vlm, tok, counts, n=1000, seed=123, dev="cuda", rows=False, max_new=3):
    """Greedy accuracy per question type, per number of objects (all types), and of "How many shapes" per number of objects.
    With rows=True the "How many shapes" replies are also split into where they go wrong (`chain_errors`)."""
    rng = np.random.default_rng(seed)
    data = []
    for i in range(n):
        objs = scene(rng, counts[i % len(counts)])
        data.append((objs, *question(rng, objs, rows), render(objs, rng)))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        replies = answer(vlm, tok, [d[3] for d in data], [d[1] for d in data], dev, max_new=max_new)
    by_kind, by_count, count_all = {}, {}, {}
    for (objs, q, a, _), r in zip(data, replies):
        by_kind.setdefault(kind(q), []).append(r == a)
        by_count.setdefault(len(objs), []).append(r == a)
        if kind(q) == "count_all":
            count_all.setdefault(len(objs), []).append(r == a)
    mean = lambda d: {k: round(float(np.mean(v)), 3) for k, v in sorted(d.items())}
    res = {"by_kind": mean(by_kind), "by_objects": mean(by_count), "count_all": mean(count_all),
           "overall": round(float(np.mean([r == d[2] for d, r in zip(data, replies)])), 3)}
    if rows:
        res["chain"] = chain_errors([(d[0], r) for d, r in zip(data, replies) if kind(d[1]) == "count_all"])
    return res


def chain_errors(pairs):
    """For (objects, row-by-row reply) pairs: is the reply well formed, are its row counts right (per true row count),
    is its arithmetic right given its own row counts, and is its final total right (per number of objects)."""
    form, arith, total, row_ok = {}, {}, {}, {}
    for objs, r in pairs:
        n, p = len(objs), parse_chain(r)
        form.setdefault(n, []).append(p is not None and len(p[0]) == GRID)
        if p is None or len(p[0]) != GRID:
            continue
        rows, sums = p
        arith.setdefault(n, []).append(sums == list(np.cumsum(rows)[1:]))
        total.setdefault(n, []).append(sums[-1] == n)
        for got, true in zip(rows, row_counts(objs)):
            row_ok.setdefault(true, []).append(got == true)
    mean = lambda d: {k: round(float(np.mean(v)), 3) for k, v in sorted(d.items())}
    return {"well_formed": mean(form), "arith_ok": mean(arith), "total_ok": mean(total), "row_ok_by_true_count": mean(row_ok)}


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
    vlm: str = ""  # start from a trained VLM checkpoint instead of a fresh ViT + projector
    rows: bool = False  # answer "How many shapes" row by row with a running sum (tasks.chain)
    drill: float = 0.0  # share of text-only sums (tasks.sum_question) in each batch
    max_new: int = 3  # reply tokens at eval; a row-by-row answer needs up to 16
    eval_max: int = 10  # eval scenes have 1..eval_max objects
    text: str = ""  # token shard replayed alongside (e.g. data/sft_train.bin) so an unfrozen LM keeps its text skills
    text_batch: int = 16  # x ctx_len tokens per step
    seed: int = 0


def build(vc, dev, vlm_ckpt=None):
    tok = chat_tokenizer(Tokenizer.load(vc.tok))  # adds <|image|> if the file predates it
    lm = load_checkpoint(vc.init, dev, tok.vocab_size)
    vit = ViT(Config(vocab_size=1, ctx_len=N_PATCHES, n_layer=vc.vit_layers, n_head=vc.vit_dim // 64, d_model=vc.vit_dim, causal=False))
    vlm = VLM(lm, vit, tok.specials[IMAGE]).to(dev)
    if vlm_ckpt:
        vlm.load_state_dict(torch.load(vlm_ckpt, map_location=dev)["vlm"])
    if vc.freeze_lm:
        vlm.lm.requires_grad_(False)
    return tok, vlm


def text_batches(c, dev):
    return batches(c.text, c.text_batch, Config.ctx_len, dev, c.seed) if c.text else None


def logger(path):
    log = open(path, "a")

    def emit(**row):
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(row, flush=True)

    return emit


def main(vc):
    dev = "cuda"
    run = os.path.join("runs", vc.name)
    os.makedirs(run, exist_ok=True)
    torch.manual_seed(vc.seed)
    tok, vlm = build(vc, dev, vc.vlm)
    emit = logger(os.path.join(run, "log.jsonl"))
    emit(event="start", trainable=sum(p.numel() for p in vlm.parameters() if p.requires_grad),
         vit=sum(p.numel() for p in vlm.vit.parameters()), config=dataclasses.asdict(vc))
    rng = np.random.default_rng(vc.seed)
    t0 = time.time()

    def every(step, loss):
        emit(step=step, loss=round(loss, 4), secs=round(time.time() - t0))
        if step % 1000 == 0 or step == vc.steps:
            vlm.eval()
            emit(step=step, eval=evaluate(vlm, tok, list(range(1, vc.eval_max + 1)), rows=vc.rows, max_new=vc.max_new))
            vlm.train()

    counts = list(range(vc.min_count, vc.max_count + 1))
    if not vc.freeze_lm:  # the LM learns too, so watch its text skills (stage A leaves them untouched)
        emit(step=0, skills=skills(vlm.lm, tok, digits=(4, 8, 12, 16)))
    fit(vlm, tok, lambda: examples(rng, vc.batch, counts, vc.rows, vc.drill), vc.steps, vc.lr, vc.warmup, vc.L, dev, every,
        text_batches(vc, dev))
    if not vc.freeze_lm:
        emit(step=vc.steps, skills=skills(vlm.lm, tok, digits=(4, 8, 12, 16)))
    torch.save({"vlm": vlm.state_dict(), "config": dataclasses.asdict(vc)}, os.path.join(run, "ckpt.pt"))
    emit(event="done", secs=round(time.time() - t0))


# ---- image RSI: the stage-5 loop, with object count as the frontier -------------------------------
#
# Same expert iteration as rsi.py (pose -> sample k -> verifier keeps -> fine-tune on own answers -> promote),
# but the training path differs (images generated in memory, not token shards), so it lives here.
# One difference in the filter: answers are short ("7", "yes", "red"), so with k=8 samples one of them is
# often right by luck. A lucky hit would just hand the model the label. The default filter keeps a problem
# only when the *majority* of its k samples is the correct answer (strictly more votes than any other reply).
# Two comparison runs: "any" (stage 5's filter: one correct sample is enough) and "oracle" (every posed
# problem with its true label, i.e. plain supervised learning on the same scenes).


@dataclass
class ImageRSIConfig:
    name: str = "vrsi"
    init: str = "runs/rsi/r10/ckpt.pt"  # the LM (frozen, as in stage A)
    vlm: str = "runs/vlm/ckpt.pt"  # stage-A VLM
    tok: str = "data/tok_chat.json"
    freeze_lm: bool = True
    vit_layers: int = 4
    vit_dim: int = 256
    trained_max: int = 6  # stage A saw 1-6 objects; the frontier starts above it
    max_count: int = GRID * GRID  # a 4x4 grid holds at most 16 objects
    rounds: int = 10
    problems: int = 40000  # per round, half at the frontier
    k: int = 8
    temperature: float = 0.7
    promote: float = 0.9  # "How many shapes" greedy accuracy at the frontier needed to move on
    min_kept: int = 200
    steps: int = 300  # x batch 128 per round
    batch: int = 128
    lr: float = 3e-4
    L: int = 96
    filter: str = "majority"  # or "any" / "oracle", see above
    rows: bool = False  # row-by-row answers (long, so lucky hits vanish and "any" is safe)
    drill: float = 0.0  # share of text-only sums in the replay half (sampled and verified like the rest)
    count_only: bool = False  # frontier problems ask only "How many shapes" (the question promotion reads)
    max_new: int = 3
    text: str = ""  # see VisionConfig
    text_batch: int = 16
    eval_n: int = 4000  # 250 scenes per object count, ~60 of them "How many shapes"
    seed: int = 1  # eval uses 123


def majority(replies, truth):
    """True when the correct answer gets strictly more votes than any other reply."""
    votes = collections.Counter(replies)
    return votes[truth] > max([v for a, v in votes.items() if a != truth], default=0)


def sample_and_filter(vlm, tok, exs, frontier, k, temperature, filter, max_new):
    """Keep problems by `filter`. Stats split by frontier/replay; "kept" counts majority-correct problems,
    "passk" any-correct ones, "lucky" those where exactly one of k samples was right."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        outs = answer(vlm, tok, [e[1] for e in exs] * k, [e[2] for e in exs] * k, "cuda", temperature, max_new)
    n = len(exs)
    stats = {s: collections.Counter() for s in ("frontier", "replay")}
    kept = []
    for i, e in enumerate(exs):
        replies = [outs[i + j * n] for j in range(k)]
        right, ok = replies.count(e[3]), majority(replies, e[3])
        st = stats["frontier" if len(e[0]) == frontier else "replay"]
        st.update(n=1, pass1=replies[0] == e[3], passk=right > 0, lucky=right == 1, kept=ok)
        if kind(e[2]) in ("count_all", "sum"):
            st.update(**{kind(e[2]): 1, kind(e[2]) + "_kept": ok, kind(e[2]) + "_any": right > 0})
        if {"majority": ok, "any": right > 0, "oracle": True}[filter]:
            kept.append(e)
    return kept, {s: {key: (round(v / st["n"], 4) if key in ("pass1", "passk", "lucky") else v) for key, v in st.items()}
                  for s, st in stats.items()}


def self_improve(rc):
    dev = "cuda"
    run = os.path.join("runs", rc.name)
    os.makedirs(run, exist_ok=True)
    emit = logger(os.path.join(run, "rsi.jsonl"))
    torch.manual_seed(rc.seed)
    tok, vlm = build(rc, dev, rc.vlm)
    counts = list(range(1, rc.max_count + 1))
    acc = evaluate(vlm, tok, counts, n=rc.eval_n, rows=rc.rows, max_new=rc.max_new)
    frontier = next((c for c in range(rc.trained_max + 1, rc.max_count + 1) if acc["count_all"][c] < rc.promote), rc.max_count)
    emit(round=0, frontier=frontier, eval=acc, config=dataclasses.asdict(rc))
    if not rc.freeze_lm:
        emit(round=0, skills=skills(vlm.lm, tok, digits=(4, 8, 12, 16)))
    rng = np.random.default_rng(rc.seed)
    text = text_batches(rc, dev)
    for r in range(1, rc.rounds + 1):
        t0 = time.time()
        exs = (examples(rng, rc.problems // 2, [frontier], rc.rows, count_only=rc.count_only)
               + examples(rng, rc.problems // 2, list(range(1, frontier)), rc.rows, rc.drill))
        kept, stats = sample_and_filter(vlm, tok, exs, frontier, rc.k, rc.temperature, rc.filter, rc.max_new)
        if stats["frontier"]["kept"] < rc.min_kept and rc.filter == "majority":
            emit(round=r, stop="too few verified frontier answers", sampling=stats)
            break
        fit(vlm, tok, lambda: [kept[i] for i in rng.integers(len(kept), size=rc.batch)], rc.steps, rc.lr, 20, rc.L, dev,
            text=text)
        acc = evaluate(vlm, tok, counts, n=rc.eval_n, rows=rc.rows, max_new=rc.max_new)
        emit(round=r, frontier=frontier, kept=len(kept), sampling=stats, eval=acc, secs=round(time.time() - t0))
        torch.save({"vlm": vlm.state_dict(), "config": dataclasses.asdict(rc)}, os.path.join(run, "ckpt.pt"))
        if acc["count_all"][frontier] >= rc.promote:
            frontier = min(frontier + 1, rc.max_count)
    if not rc.freeze_lm:
        emit(round=r, skills=skills(vlm.lm, tok, digits=(4, 8, 12, 16)))


if __name__ == "__main__":
    rsi = sys.argv[1:2] == ["rsi"]
    cls = ImageRSIConfig if rsi else VisionConfig
    ap = argparse.ArgumentParser(description="Image RSI (count past the trained range)" if rsi else "Train the VLM on synthetic scenes")
    for f in dataclasses.fields(cls):
        typ = (lambda s: s.lower() in ("1", "true", "yes")) if isinstance(f.default, bool) else type(f.default)
        ap.add_argument(f"--{f.name}", type=typ, default=f.default)
    (self_improve if rsi else main)(cls(**vars(ap.parse_args(sys.argv[2:] if rsi else sys.argv[1:]))))

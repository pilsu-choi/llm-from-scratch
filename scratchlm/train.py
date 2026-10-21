"""Pretraining loop: bf16 autocast, gradient accumulation, cosine LR, eval, samples, resumable checkpoints.

    scripts/pgx.sh python -m scratchlm.train --name small --steps 15000
Writes runs/<name>/log.jsonl (one JSON object per line) and runs/<name>/ckpt.pt (latest, atomic replace).
"""
import argparse
import dataclasses
import json
import math
import os
import time
from dataclasses import dataclass

import torch

from .data import batches
from .model import GPT, Config, load_checkpoint
from .optim import AdamW, clip_grad_norm, cosine_lr
from .tokenizer import Tokenizer

PEAK_FLOPS = 68e12  # bf16 matmul measured on the GB10 in stage 0; MFU is relative to this


@dataclass
class TrainConfig:
    name: str = "small"
    train: str = "data/ts_train.bin"  # comma-separated shards are mixed in equal shares (data.batches)
    val: str = "data/ts_val.bin"  # comma-separated: one val loss per shard, `val` logs the first
    tok: str = "data/tok8k.json"
    init: str = ""  # start from these weights (e.g. pretrained ckpt for SFT); vocab grows to the tokenizer's
    batch: int = 64  # sequences per micro-batch
    accum: int = 1  # micro-batches per optimizer step
    steps: int = 15000
    lr: float = 1e-3
    warmup: int = 500
    min_lr_ratio: float = 0.1
    wd: float = 0.1
    clip: float = 1.0
    eval_every: int = 500
    eval_batches: int = 20
    ckpt_every: int = 1000
    prompts: str = "Once upon a time"  # samples at each eval, "|"-separated
    compile: bool = True
    seed: int = 0


def parse():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for f in dataclasses.fields(TrainConfig) + dataclasses.fields(Config):
        if f.type is bool or isinstance(f.default, bool):
            ap.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"), default=f.default)
        else:
            ap.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    a = vars(ap.parse_args())
    pick = lambda cls: cls(**{f.name: a[f.name] for f in dataclasses.fields(cls)})
    return pick(TrainConfig), pick(Config)


@torch.no_grad()
def evaluate(model, val_batches):
    losses = []
    for x, y in val_batches:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses.append(model(x, y)[1].item())
    return sum(losses) / len(losses)


def main(tc=None, mc=None):
    if tc is None:
        tc, mc = parse()
    dev = "cuda"
    run = os.path.join("runs", tc.name)
    os.makedirs(run, exist_ok=True)
    tok = Tokenizer.load(tc.tok)
    torch.manual_seed(tc.seed)
    if tc.init:
        model = load_checkpoint(tc.init, dev, tok.vocab_size)
        mc = model.c
    else:
        mc.vocab_size = tok.vocab_size
        model = GPT(mc).to(dev)
    opt = AdamW(model.parameters(), weight_decay=tc.wd)
    n_params = sum(p.numel() for p in model.parameters())

    ckpt_path, start = os.path.join(run, "ckpt.pt"), 0
    if os.path.exists(ckpt_path):  # resume
        ck = torch.load(ckpt_path, map_location=dev)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start = ck["step"]
    log = open(os.path.join(run, "log.jsonl"), "a")

    def emit(**row):
        log.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.flush()
        print(row, flush=True)

    tokens_per_step = tc.batch * tc.accum * mc.ctx_len
    flops_per_token = 6 * n_params + 12 * mc.n_layer * mc.ctx_len * mc.d_model  # matmuls fwd+bwd, + attention scores
    if start == 0:
        emit(event="start", params=n_params, tokens_per_step=tokens_per_step, train=dataclasses.asdict(tc), model=dataclasses.asdict(mc))
    else:
        emit(event="resume", step=start)

    train = batches(tc.train, tc.batch, mc.ctx_len, dev, seed=tc.seed + start)  # new seed on resume: no repeated windows
    vals = {}  # fixed val sets: comparable across steps and runs
    for p in tc.val.split(","):
        g = batches(p, tc.batch, mc.ctx_len, dev, seed=12345)
        vals[p] = [next(g) for _ in range(tc.eval_batches)]
    fwd = torch.compile(model) if tc.compile else model
    prompts = [torch.tensor([tok.encode(p)], device=dev) for p in tc.prompts.split("|")]

    def save(step):
        tmp = ckpt_path + ".tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "model_config": dataclasses.asdict(mc)}, tmp)
        os.replace(tmp, ckpt_path)  # atomic: a crash never leaves a half-written checkpoint

    t0, loss_sum = time.time(), torch.zeros((), device=dev)
    for step in range(start, tc.steps):
        lr = cosine_lr(step, tc.lr, tc.warmup, tc.steps, tc.min_lr_ratio)
        for _ in range(tc.accum):
            x, y = next(train)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = fwd(x, y)
            (loss / tc.accum).backward()  # grads add up over micro-batches = one big batch
            loss_sum += loss.detach() / tc.accum
        norm = clip_grad_norm(opt.params, tc.clip)
        opt.step(lr)
        opt.zero_grad()

        done = step + 1
        if done % 50 == 0:
            torch.cuda.synchronize()
            dt = time.time() - t0
            tps = 50 * tokens_per_step / dt
            emit(step=done, loss=round(loss_sum.item() / 50, 4), lr=lr, grad_norm=round(norm.item(), 3),
                 tok_s=round(tps), mfu=round(tps * flops_per_token / PEAK_FLOPS, 3))
            t0, loss_sum = time.time(), torch.zeros((), device=dev)
        if done % tc.eval_every == 0 or done == tc.steps:
            losses = {p: round(evaluate(fwd, v), 4) for p, v in vals.items()}
            samples = [tok.decode(model.generate(p, 120, temperature=0.8, top_k=50)[0].tolist()) for p in prompts]
            emit(step=done, val=next(iter(losses.values())), **({"vals": losses} if len(vals) > 1 else {}),
                 sample=samples[0], **({"samples": samples} if len(samples) > 1 else {}))
            t0 = time.time()  # don't count eval time in throughput
        if done % tc.ckpt_every == 0 or done == tc.steps:
            save(done)
    emit(event="done", step=tc.steps)


if __name__ == "__main__":
    main()

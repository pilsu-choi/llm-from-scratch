"""Recursive self-improvement on addition: expert iteration (STaR / ReST-EM) with a program verifier.

Each round:
  1. pose problems: half at the frontier (the longer operand has D digits), half below it (replay)
  2. the current model samples k answers per problem
  3. the verifier keeps one correct answer per problem; wrong ones are dropped
  4. fine-tune the current model on its own kept answers -> next model
  5. measure greedy accuracy by digits; if the frontier is solved well enough, D += 1
No answer in the training data comes from anywhere but the model; ground truth only grades.

    DETACH=rsi scripts/pgx.sh python -m scratchlm.rsi --init runs/sft_pad/ckpt.pt --rounds 10
"""
import argparse
import dataclasses
import json
import os
import time
from dataclasses import dataclass

import numpy as np

from .model import Config, load_checkpoint
from .sft import write_shard
from .tasks import TEMPLATES, WIDTH, addition, addition_accuracy, check_addition, reply
from .tokenizer import Tokenizer
from .train import TrainConfig
from .train import main as train


@dataclass
class RSIConfig:
    name: str = "rsi"
    init: str = "runs/sft_pad/ckpt.pt"
    tok: str = "data/tok_chat.json"
    lang: str = "en"  # prompt language of the posed problems ("ko": stage 8); the other language is only measured
    mix: str = ""  # comma-separated SFT shards trained alongside each round's kept answers (replay against forgetting)
    val: str = "data/sft_val.bin"
    start_digits: int = 8  # lowest possible frontier (SFT trained operands of 1-7 digits)
    max_digits: int = WIDTH
    rounds: int = 10
    problems: int = 40000  # per round
    k: int = 8  # samples per problem
    min_kept: int = 200  # stop when fewer frontier problems get a verified answer
    temperature: float = 0.7
    promote: float = 0.9  # frontier greedy accuracy needed to move on
    steps: int = 150  # fine-tuning steps per round (x 32,768 tokens; ~2 passes over the kept answers)
    lr: float = 1e-4
    seed: int = 0


def pose(rng, n, frontier, lang="en"):
    """Half the problems at the frontier (longer operand has `frontier` digits), half below it (replay),
    grouped by operand lengths for per-group stats."""
    groups = {}
    for i in range(n):
        top = frontier if i % 2 == 0 or frontier == 1 else int(rng.integers(1, frontier))
        da, db = top, int(rng.integers(1, top + 1))
        if rng.random() < 0.5:
            da, db = db, da
        groups.setdefault((da, db), []).append(addition(rng, da, db, lang=lang)[0])
    return groups


def sample_and_filter(model, tok, groups, frontier, k, temperature):
    """Self-generated training data: for each problem, the first of k sampled answers the verifier accepts.
    Stats are split into frontier vs replay problems: averaged 50/50, a ~99% replay half would make a 0% frontier look like ~50%."""
    kept, stats = [], {s: {"n": 0, "pass1": 0, "passk": 0, "kept": 0} for s in ("frontier", "replay")}
    for (da, db), probs in groups.items():
        st = stats["frontier" if max(da, db) == frontier else "replay"]
        outs = reply(model, tok, [p for p in probs for _ in range(k)], WIDTH + 2, temperature)
        for i, p in enumerate(probs):
            ok = [o for o in outs[i * k : (i + 1) * k] if check_addition(p, o)]
            st["n"] += 1
            st["pass1"] += check_addition(p, outs[i * k])
            st["passk"] += bool(ok)
            if ok:
                st["kept"] += 1
                kept.append((p, ok[0]))
    for st in stats.values():
        st["pass1"], st["passk"] = round(st["pass1"] / max(1, st["n"]), 4), round(st["passk"] / max(1, st["n"]), 4)
    return kept, stats


def main(rc):
    run = os.path.join("runs", rc.name)
    os.makedirs(run, exist_ok=True)
    log = open(os.path.join(run, "rsi.jsonl"), "a")
    tok = Tokenizer.load(rc.tok)
    rng = np.random.default_rng(rc.seed)
    prev = rc.init
    model = load_checkpoint(prev).eval()
    digits = range(1, rc.max_digits + 1)
    measure = lambda: {lang: addition_accuracy(model, tok, digits, lang=lang) for lang in TEMPLATES}  # both languages, always
    accs = measure()
    acc = accs[rc.lang]
    # start where the model is not yet reliable (SFT already generalizes a few digits past what it trained on)
    frontier = next((d for d in range(rc.start_digits, rc.max_digits + 1) if acc[d] < rc.promote), rc.max_digits)
    row = {"round": 0, "frontier": frontier, "accuracy": acc, "accuracy_by_lang": accs}
    log.write(json.dumps(row) + "\n")
    print(row, flush=True)
    for r in range(1, rc.rounds + 1):
        t0 = time.time()
        kept, stats = sample_and_filter(model, tok, pose(rng, rc.problems, frontier, rc.lang), frontier, rc.k, rc.temperature)
        if stats["frontier"]["kept"] < rc.min_kept:  # nothing new to learn from: training would be pure replay
            print({"round": r, "stop": "too few verified frontier answers", "stats": stats}, flush=True)
            break
        shard = os.path.join(run, f"round{r}")
        n_tokens, _ = write_shard(tok, kept, shard)
        assert n_tokens > 4 * Config.ctx_len, "shard too small for training windows"
        train(  # fine-tune the current model on its own verified answers
            TrainConfig(name=f"{rc.name}/r{r}", init=prev, tok=rc.tok, train=",".join([shard + ".bin", *filter(None, rc.mix.split(","))]), val=rc.val,
                        steps=rc.steps, lr=rc.lr, warmup=30, eval_every=rc.steps, ckpt_every=rc.steps, seed=rc.seed + r),
            Config(),
        )
        prev = os.path.join(run, f"r{r}", "ckpt.pt")
        model = load_checkpoint(prev).eval()
        accs = measure()
        acc = accs[rc.lang]
        row = {"round": r, "frontier": frontier, "sampling": stats, "tokens": n_tokens, "accuracy": acc, "accuracy_by_lang": accs, "secs": round(time.time() - t0)}
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(row, flush=True)
        if acc[frontier] >= rc.promote:
            frontier = min(frontier + 1, rc.max_digits)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Recursive self-improvement on addition")
    for f in dataclasses.fields(RSIConfig):
        ap.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    main(RSIConfig(**vars(ap.parse_args())))

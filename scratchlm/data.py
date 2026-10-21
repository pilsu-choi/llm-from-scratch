"""Token shards: text -> uint16 token ids on disk, then random training windows read via np.memmap.

A shard is one flat array of ids: doc_1 <eot> doc_2 <eot> ... Training samples random (ctx_len + 1)-token
windows from it (input = window[:-1], target = window[1:]). Windows may cross <eot>; the model learns
that a new story starts after it. uint16 fits vocab_size <= 65536 and halves disk/RAM vs int32.
"""
from multiprocessing import Pool

import numpy as np
import torch

from .tokenizer import Tokenizer

EOT = "<|endoftext|>"
_tok = None


def _init(tok_path):
    global _tok
    _tok = Tokenizer.load(tok_path)


def _encode(docs):
    eot = _tok.specials[EOT]
    ids = []
    for d in docs:
        ids += _tok.encode(d)
        ids.append(eot)
    return np.array(ids, dtype=np.uint16)


def prepare(txt_path, tok_path, out_path, workers=16):
    """Encode every <|endoftext|>-separated document in parallel and append the ids to out_path."""
    with open(txt_path, encoding="utf-8") as f:
        docs = [d.strip() for d in f.read().split(EOT)]
    docs = [d for d in docs if d]
    chunks = [docs[i : i + 2000] for i in range(0, len(docs), 2000)]
    n = 0
    with Pool(workers, _init, (tok_path,)) as pool, open(out_path, "wb") as out:
        for arr in pool.imap(_encode, chunks):  # imap keeps document order
            arr.tofile(out)
            n += len(arr)
    in_bytes = sum(len(d.encode()) for d in docs)
    print(f"{out_path}: {len(docs):,} docs, {n:,} tokens, {in_bytes / n:.2f} bytes/token")


def batches(path, batch, ctx_len, device, seed=0):
    """Endless (x, y) int64 batches of random windows from a uint16 shard."""
    data = np.memmap(path, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(seed)
    while True:
        starts = rng.integers(0, len(data) - ctx_len - 1, batch)
        w = torch.from_numpy(np.stack([data[s : s + ctx_len + 1] for s in starts]).astype(np.int64))
        if device == "cuda":
            w = w.pin_memory().to(device, non_blocking=True)  # async host->GPU copy
        yield w[:, :-1], w[:, 1:]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Encode a text file into a uint16 token shard")
    ap.add_argument("txt")
    ap.add_argument("tok")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    prepare(a.txt, a.tok, a.out, a.workers)

"""SFT data: TinyStories-Instruct stories + 1-7 digit addition, packed into masked uint16 shards.

    python -m scratchlm.sft prepare      # data/sft_{train,val}.bin + .mask, data/tok_chat.json
    python -m scratchlm.sft eval runs/<name>/ckpt.pt
Training reuses scratchlm.train with --init <pretrained ckpt> --train data/sft_train.bin.
"""
import json
import random

import numpy as np
import torch

from .model import load_checkpoint
from .tasks import EOT, addition, addition_accuracy, chat_tokenizer, encode_chat, parse_instruct, reply, words_used
from .tokenizer import Tokenizer

SFT_DIGITS = tuple(range(1, 8))  # operand lengths trained on; 8+ digits are held out (the RSI frontier)


def examples(instruct_path, n_add, seed):
    with open(instruct_path, encoding="utf-8") as f:
        ex = [e for e in map(parse_instruct, f.read().split(EOT)) if e]
    n_story = len(ex)
    rng = np.random.default_rng(seed)
    # operand lengths drawn independently; half the problems shifted so every digit position is exercised
    ex += [addition(rng, *rng.choice(SFT_DIGITS, 2), shift=i % 2 == 1) for i in range(n_add)]
    random.Random(seed).shuffle(ex)
    return ex, n_story


def write_shard(tok, ex, out):
    ids, mask = [], []
    for prompt, answer in ex:
        i, m = encode_chat(tok, prompt, answer)
        ids += i
        mask += m
    np.array(ids, dtype=np.uint16).tofile(out + ".bin")
    np.array(mask, dtype=np.uint8).tofile(out + ".mask")
    return len(ids), sum(mask)


def prepare(tok_path="data/tok8k.json", seed=0):
    tok = chat_tokenizer(Tokenizer.load(tok_path))
    tok.save("data/tok_chat.json")
    for split, src, n_add in (("train", "data/TinyStories-Instruct-train200M.txt", 1_500_000), ("val", "data/TinyStories-Instruct-valid.txt", 15_000)):
        ex, n_story = examples(src, n_add, seed)
        n, n_trained = write_shard(tok, ex, f"data/sft_{split}")
        print(f"sft_{split}: {n_story:,} stories + {n_add:,} additions, {n:,} tokens ({n_trained / n:.0%} trained on)")


def evaluate(ckpt_path, tok_path="data/tok_chat.json", n_story=100):
    tok = Tokenizer.load(tok_path)
    model = load_checkpoint(ckpt_path, vocab_size=tok.vocab_size).eval()  # a pretrained ckpt gets the role tokens too
    res = {"addition": addition_accuracy(model, tok)}
    with open("data/TinyStories-Instruct-valid.txt", encoding="utf-8") as f:
        stories = [e for e in map(parse_instruct, f.read().split(EOT)) if e and words_used(e[0], "") is not None][:n_story]
    torch.manual_seed(0)
    replies = [reply(model, tok, [p], 300, temperature=0.8)[0] for p, _ in stories]
    res["story_words_used"] = round(sum(words_used(p, r) for (p, _), r in zip(stories, replies)) / len(stories), 3)
    res["example"] = {"prompt": stories[0][0], "reply": replies[0]}
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return res


if __name__ == "__main__":
    import sys

    prepare() if sys.argv[1] == "prepare" else evaluate(*sys.argv[2:])

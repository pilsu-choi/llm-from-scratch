"""SFT data: TinyStories-Instruct stories + 1-7 digit addition, packed into masked uint16 shards.

    python -m scratchlm.sft prepare      # data/sft_{train,val}.bin + .mask, data/tok_chat.json
    python -m scratchlm.sft prepare_ko   # stage 8: data/sft_{en_story,ko_story,ko_add}_{train,val} with data/tok_ko.json
    python -m scratchlm.sft eval runs/<name>/ckpt.pt [tok.json] [en,ko]
Training reuses scratchlm.train with --init <pretrained ckpt> --train data/sft_train.bin.
"""
import json
import random

import numpy as np
import torch

from .model import load_checkpoint
from .tasks import EOT, addition, addition_accuracy, chat_tokenizer, continue_story, encode_chat, parse_instruct, reply, words_used
from .tokenizer import Tokenizer

SFT_DIGITS = tuple(range(1, 8))  # operand lengths trained on; 8+ digits are held out (the RSI frontier)
KO_DIGITS = tuple(range(1, 4))  # stage 8: Korean-worded addition is trained on 1-3 digits only (4+ must transfer)


def examples(path, n_add, seed, story=parse_instruct, digits=SFT_DIGITS, lang="en", max_stories=None):
    with open(path, encoding="utf-8") as f:
        ex = [e for e in map(story, f.read().split(EOT)[:max_stories]) if e]
    n_story = len(ex)
    rng = np.random.default_rng(seed)
    # operand lengths drawn independently; half the problems shifted so every digit position is exercised
    ex += [addition(rng, *rng.choice(digits, 2), shift=i % 2 == 1, lang=lang) for i in range(n_add)]
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


def prepare_ko(seed=0):
    """Stage 8 pieces, mixed 1:1 at train time (`--train a.bin,b.bin`) so the three runs differ in one piece each:
    English stories (+ English addition = the existing sft_train), Korean stories (+ Korean addition 1-3 digits).
    The existing sft_{train,val} shards are reused as-is: tok_ko encodes English exactly like tok_chat."""
    tok = Tokenizer.load("data/tok_ko.json")
    instruct = {"train": "data/TinyStories-Instruct-train200M.txt", "val": "data/TinyStories-Instruct-valid.txt"}
    korean = {"train": "data/TinyStories-Korean-train.txt", "val": "data/TinyStories-Korean-valid.txt"}
    n_ko = {"train": 200_000, "val": 2_000}  # about as many stories as the English Instruct file has
    n_add = {"train": 500_000, "val": 5_000}  # Korean shard ~ half addition tokens, like the English one
    for split in ("train", "val"):
        pieces = {
            "en_story": examples(instruct[split], 0, seed),
            "ko_story": examples(korean[split], 0, seed, continue_story, max_stories=n_ko[split]),
            "ko_add": examples(korean[split], n_add[split], seed, continue_story, KO_DIGITS, "ko", n_ko[split]),
        }
        for name, (ex, n_story) in pieces.items():
            n, n_trained = write_shard(tok, ex, f"data/sft_{name}_{split}")
            print(f"sft_{name}_{split}: {n_story:,} stories + {len(ex) - n_story:,} additions, {n:,} tokens ({n_trained / n:.0%} trained on)")


def evaluate(ckpt_path, tok_path="data/tok_chat.json", langs="en", n_story=100):
    tok = Tokenizer.load(tok_path)
    model = load_checkpoint(ckpt_path, vocab_size=tok.vocab_size).eval()  # a pretrained ckpt gets the role tokens too
    res = skills(model, tok, int(n_story), langs=tuple(langs.split(",")))
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return res


@torch.no_grad()
def skills(model, tok, n_story=100, digits=range(1, 17), langs=("en",)):
    """The two text skills: greedy addition accuracy per digit count (per prompt language), and the share of required story words used."""
    res = {"addition" if lang == "en" else f"addition_{lang}": addition_accuracy(model, tok, digits, lang=lang) for lang in langs}
    with open("data/TinyStories-Instruct-valid.txt", encoding="utf-8") as f:
        stories = [e for e in map(parse_instruct, f.read().split(EOT)) if e and words_used(e[0], "") is not None][:n_story]
    torch.manual_seed(0)
    replies = [reply(model, tok, [p], 300, temperature=0.8)[0] for p, _ in stories]
    res["story_words_used"] = round(sum(words_used(p, r) for (p, _), r in zip(stories, replies)) / len(stories), 3)
    res["example"] = {"prompt": stories[0][0], "reply": replies[0]}
    return res


if __name__ == "__main__":
    import sys

    {"prepare": prepare, "prepare_ko": prepare_ko}.get(sys.argv[1], lambda: evaluate(*sys.argv[2:]))()

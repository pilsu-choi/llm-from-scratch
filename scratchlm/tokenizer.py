"""Byte-level BPE tokenizer, from scratch (stdlib only).

Pipeline:  text -> pre-tokenize (regex chunks) -> UTF-8 bytes -> apply merges -> ids
Byte-level means the base vocab is the 256 byte values, so any string (English,
Korean, emoji) is encodable with no <unk>. Training greedily merges the most
frequent adjacent pair, counted over unique chunks weighted by frequency.
"""
import json
import re
import sys
from collections import Counter, defaultdict

# GPT-2 style split using stdlib `re`: contractions, words (with leading space),
# 1-3 digit groups, punctuation runs, whitespace. Merges never cross chunk borders.
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d{1,3}| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+""")


def merge(ids, pair, new):
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class Tokenizer:
    def __init__(self, merges=None, specials=None):
        self.merges = merges or {}  # (a, b) -> new id; insertion order == rank
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), new in self.merges.items():
            self.vocab[new] = self.vocab[a] + self.vocab[b]
        self.specials = specials or {}  # "<|eos|>" -> id
        for s, i in self.specials.items():
            self.vocab[i] = s.encode()
        self._split = re.compile("(" + "|".join(map(re.escape, self.specials)) + ")") if self.specials else None
        self._cache = {}

    @property
    def vocab_size(self):
        return len(self.vocab)

    @classmethod
    def train(cls, texts, vocab_size, specials=(), verbose=False):
        counts = Counter(c for t in texts for c in PAT.findall(t))
        words = [list(w.encode()) for w in counts]
        freqs = list(counts.values())
        stats, where = Counter(), defaultdict(set)  # pair -> weighted count / word indices
        for i, w in enumerate(words):
            for p in zip(w, w[1:]):
                stats[p] += freqs[i]
                where[p].add(i)

        merges, vocab = {}, {i: bytes([i]) for i in range(256)}
        for new in range(256, vocab_size - len(specials)):
            if not stats:
                break
            pair = max(stats, key=stats.get)
            merges[pair] = new
            vocab[new] = vocab[pair[0]] + vocab[pair[1]]
            for i in where.pop(pair):  # only words containing the pair change
                w, f = words[i], freqs[i]
                for p in zip(w, w[1:]):
                    stats[p] -= f
                    if stats[p] <= 0:
                        del stats[p]
                words[i] = w = merge(w, pair, new)
                for p in zip(w, w[1:]):
                    stats[p] += f
                    where[p].add(i)
            if verbose and new % 500 == 0:
                print(f"merge {new}: {pair} -> {vocab[new]!r}", file=sys.stderr)
        n = 256 + len(merges)
        return cls(merges, {s: n + k for k, s in enumerate(specials)})

    def _encode_chunk(self, chunk):
        if chunk in self._cache:
            return self._cache[chunk]
        ids = list(chunk.encode())
        while len(ids) > 1:  # apply the earliest-learned merge first, like training did
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        self._cache[chunk] = ids
        return ids

    def encode(self, text):
        parts = self._split.split(text) if self._split else [text]
        out = []
        for part in parts:
            if part in self.specials:
                out.append(self.specials[part])
            else:
                for c in PAT.findall(part):
                    out += self._encode_chunk(c)
        return out

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"merges": [[a, b, n] for (a, b), n in self.merges.items()], "specials": self.specials}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls({(a, b): n for a, b, n in d["merges"]}, d["specials"])


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer")
    ap.add_argument("input")
    ap.add_argument("out")
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--max-mb", type=float, default=50, help="train on the first N MB of the input")
    ap.add_argument("--specials", default="<|endoftext|>")
    a = ap.parse_args()
    with open(a.input, encoding="utf-8") as f:
        text = f.read(int(a.max_mb * 2**20))
    specials = a.specials.split(",")
    docs = re.split("|".join(map(re.escape, specials)), text)  # never learn merges across specials
    tok = Tokenizer.train(docs, a.vocab, specials, verbose=True)
    tok.save(a.out)
    sample = docs[1][:300]
    ids = tok.encode(sample)
    print(f"vocab={tok.vocab_size} sample: {len(sample.encode())} bytes -> {len(ids)} tokens")
    print(" | ".join(tok.decode([i]) for i in ids[:40]))

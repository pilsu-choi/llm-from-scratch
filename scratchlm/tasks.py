"""Chat template, the tasks we fine-tune on, and their program verifiers (also used by RSI in stage 5).

Template:  <|user|>{prompt}<|assistant|>{answer}<|endoftext|>
Only the answer and the closing <|endoftext|> are trained on (mask=1); the prompt is context (mask=0).
"""
import re

import numpy as np

from .tokenizer import Tokenizer

USER, ASSISTANT, EOT = "<|user|>", "<|assistant|>", "<|endoftext|>"


def chat_tokenizer(tok):
    """Same merges, plus the two role tokens appended after the existing ids (the model's vocab grows by 2)."""
    new = [s for s in (USER, ASSISTANT) if s not in tok.specials]
    return Tokenizer(tok.merges, {**tok.specials, **{s: tok.vocab_size + i for i, s in enumerate(new)}})


def encode_chat(tok, prompt, answer):
    """Token ids and a train mask (1 = predict this token) for one example."""
    p = tok.encode(USER + prompt + ASSISTANT)
    a = tok.encode(answer + EOT)
    return p + a, [0] * len(p) + [1] * len(a)


# ---- addition: verifiable by construction --------------------------------------------------------


WIDTH = 16  # every operand is zero-padded to 16 digits; the answer is always WIDTH + 1 digits


def _number(rng, d):
    return int(rng.integers(10 ** (d - 1) if d > 1 else 0, 10**d))  # d significant digits


def _answer(a, b):
    return f"{a + b:0{WIDTH + 1}d}"[::-1]


def addition(rng, da, db=None, shift=False):
    """A random da-digit + db-digit problem in a fixed-width format:
    - operands zero-padded to WIDTH digits, so the k-th digit of every number sits at the same position
      whatever its value (a model that finds digits by position never meets an unseen distance)
    - answer padded to WIDTH + 1 digits and written reversed (ones digit first), so each output digit needs
      only digits already seen plus one carry, in the order carries propagate.
    - shift=True moves each operand to a random position (1354 -> 1354000000), so every position sees
      real digits in training; without it, positions past the longest trained length are always 0 and
      the model learns to ignore them.
    See wiki/lessons/2026-10-21-length-generalization.md for why plain and reversed formats failed."""
    db = da if db is None else db
    a, b = _number(rng, da), _number(rng, db)
    if shift:
        a, b = a * 10 ** int(rng.integers(0, WIDTH - da + 1)), b * 10 ** int(rng.integers(0, WIDTH - db + 1))
    return f"What is {a:0{WIDTH}d} + {b:0{WIDTH}d}?", _answer(a, b)


def check_addition(prompt, reply):
    """Verifier: the reply must be exactly the padded, reversed sum."""
    a, b = map(int, re.findall(r"\d+", prompt))
    return reply.strip() == _answer(a, b)


# ---- story instructions (TinyStories-Instruct) ---------------------------------------------------

FIELDS = ("Words", "Features", "Summary", "Random sentence")
_FIELD = re.compile(rf"^({'|'.join(FIELDS)}): ?(.*)$")


def parse_instruct(doc):
    """Split one TinyStories-Instruct record into (prompt, story). Fields may come before or after the story."""
    fields, story, in_story = {}, [], False
    for line in doc.strip().split("\n"):
        m = _FIELD.match(line)
        if m:
            fields[m.group(1)], in_story = m.group(2).strip(), False
        elif line.startswith("Story:"):
            in_story = True
        elif in_story:
            story.append(line)
    story = "\n".join(story).strip()
    if not story:
        return None
    prompt = "\n".join(["Write a short story."] + [f"{k}: {fields[k]}" for k in FIELDS if fields.get(k)])
    return prompt, story


def words_used(prompt, reply):
    """Verifier for the story task: fraction of the required 'Words:' that appear in the reply."""
    m = re.search(r"^Words: (.*)$", prompt, re.M)
    if not m:
        return None
    words = [w.strip().lower() for w in m.group(1).split(",") if w.strip()]
    return sum(w in reply.lower() for w in words) / len(words)


# ---- evaluation ----------------------------------------------------------------------------------


def reply(model, tok, prompts, max_new, temperature=0.0, chunk=2048):
    """Generate replies for prompts of equal token length (batched, `chunk` rows at a time), cut at <|endoftext|>."""
    import torch  # verifiers above stay torch-free

    ids = [tok.encode(USER + p + ASSISTANT) for p in prompts]
    assert len({len(i) for i in ids}) == 1, "batch prompts must have the same token length"
    dev, eot, res = next(model.parameters()).device, tok.specials[EOT], []
    for c in range(0, len(ids), chunk):
        x = torch.tensor(ids[c : c + chunk], device=dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):  # ~2x faster sampling
            out = model.generate(x, max_new, temperature=temperature, top_k=50 if temperature else None)
        for row in out[:, x.shape[1] :].tolist():
            res.append(tok.decode(row[: row.index(eot)] if eot in row else row))
    return res


def addition_accuracy(model, tok, digits=range(1, WIDTH + 1), n=200, seed=0):
    """Greedy pass@1 for d-digit + d-digit problems, per d. Fixed width: every prompt has the same token length."""
    rng = np.random.default_rng(seed)
    acc = {}
    for d in digits:
        probs = [addition(rng, d)[0] for _ in range(n)]
        outs = reply(model, tok, probs, max_new=WIDTH + 2)
        acc[d] = sum(check_addition(p, o) for p, o in zip(probs, outs)) / n
    return acc

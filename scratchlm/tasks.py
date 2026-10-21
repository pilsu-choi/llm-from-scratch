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


def addition(rng, n_digits):
    """A random n-digit + n-digit problem (no leading zeros). Returns (prompt, answer)."""
    lo, hi = 10 ** (n_digits - 1) if n_digits > 1 else 0, 10**n_digits
    a, b = rng.integers(lo, hi, 2)
    return f"What is {a} + {b}?", str(a + b)


def check_addition(prompt, reply):
    """Verifier: the reply must be exactly the integer sum."""
    a, b = map(int, re.findall(r"\d+", prompt))
    return reply.strip() == str(a + b)


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


def reply(model, tok, prompts, max_new, temperature=0.0):
    """Generate replies for prompts of equal token length (batched), cut at <|endoftext|>."""
    import torch  # verifiers above stay torch-free

    ids = [tok.encode(USER + p + ASSISTANT) for p in prompts]
    assert len({len(i) for i in ids}) == 1, "batch prompts must have the same token length"
    dev = next(model.parameters()).device
    out = model.generate(torch.tensor(ids, device=dev), max_new, temperature=temperature, top_k=50 if temperature else None)
    eot = tok.specials[EOT]
    res = []
    for row in out[:, len(ids[0]) :].tolist():
        res.append(tok.decode(row[: row.index(eot)] if eot in row else row))
    return res


def addition_accuracy(model, tok, digits=range(1, 7), n=200, seed=0):
    """Greedy pass@1 by digit count. All n-digit problems tokenize to the same length, so each digit count is one batch."""
    rng = np.random.default_rng(seed)
    acc = {}
    for d in digits:
        probs = [addition(rng, d)[0] for _ in range(n)]
        outs = reply(model, tok, probs, max_new=d + 3)
        acc[d] = sum(check_addition(p, o) for p, o in zip(probs, outs)) / n
    return acc

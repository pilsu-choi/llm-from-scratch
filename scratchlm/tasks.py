"""Chat template, the tasks we fine-tune on, and their program verifiers (also used by RSI in stage 5).

Template:  <|user|>{prompt}<|assistant|>{answer}<|endoftext|>
Only the answer and the closing <|endoftext|> are trained on (mask=1); the prompt is context (mask=0).
"""
import re

import numpy as np

from .tokenizer import Tokenizer

USER, ASSISTANT, EOT, IMAGE = "<|user|>", "<|assistant|>", "<|endoftext|>", "<|image|>"


def chat_tokenizer(tok):
    """Same merges, plus the role tokens and the image placeholder appended after the existing ids (vocab grows)."""
    new = [s for s in (USER, ASSISTANT, IMAGE) if s not in tok.specials]
    return Tokenizer(tok.merges, {**tok.specials, **{s: tok.vocab_size + i for i, s in enumerate(new)}})


def encode_chat(tok, prompt, answer):
    """Token ids and a train mask (1 = predict this token) for one example."""
    p = tok.encode(USER + prompt + ASSISTANT)
    a = tok.encode(answer + EOT)
    return p + a, [0] * len(p) + [1] * len(a)


# ---- addition: verifiable by construction --------------------------------------------------------


WIDTH = 16  # every operand is zero-padded to 16 digits; the answer is always WIDTH + 1 digits
# Same numbers and answer in both languages; only the words around them change. The Korean template attaches
# no particle that depends on how the last digit is read (은/는), and neither template contains other digits.
TEMPLATES = {"en": "What is {a} + {b}?", "ko": "{a} 더하기 {b}의 값은?"}


def _number(rng, d):
    return int(rng.integers(10 ** (d - 1) if d > 1 else 0, 10**d))  # d significant digits


def _answer(a, b):
    return f"{a + b:0{WIDTH + 1}d}"[::-1]


def addition(rng, da, db=None, shift=False, lang="en"):
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
    return TEMPLATES[lang].format(a=f"{a:0{WIDTH}d}", b=f"{b:0{WIDTH}d}"), _answer(a, b)


def check_addition(prompt, reply):
    """Verifier: the reply must be exactly the padded, reversed sum."""
    a, b = map(int, re.findall(r"\d+", prompt))
    return reply.strip() == _answer(a, b)


# ---- synthetic scenes: images whose answers are known by construction ---------------------------

COLORS = {"red": (220, 50, 50), "green": (50, 190, 70), "blue": (60, 90, 230), "yellow": (235, 215, 50), "purple": (160, 70, 200)}
SHAPES = ("circle", "square", "triangle")
IMG, GRID = 64, 4  # 64x64 pixels, shapes placed in distinct cells of a 4x4 grid (16 px each, so they never overlap)


def scene(rng, n):
    """n objects, each (shape, color, row, col) in distinct grid cells."""
    cells = rng.choice(GRID * GRID, n, replace=False)
    return [(str(rng.choice(SHAPES)), str(rng.choice(list(COLORS))), int(c) // GRID, int(c) % GRID) for c in cells]


def render(objs, rng):
    """uint8 (IMG, IMG, 3) black canvas; each shape ~10-13 px with a little jitter inside its cell."""
    img = np.zeros((IMG, IMG, 3), np.uint8)
    yy, xx = np.mgrid[:IMG, :IMG]
    cell = IMG // GRID
    for shape, color, r, c in objs:
        h = rng.integers(5, 7)  # half size
        cy, cx = r * cell + cell // 2 + rng.integers(-1, 2), c * cell + cell // 2 + rng.integers(-1, 2)
        dy, dx = yy - cy, xx - cx
        if shape == "circle":
            m = dy**2 + dx**2 <= h * h
        elif shape == "square":
            m = (abs(dy) <= h - 1) & (abs(dx) <= h - 1)
        else:  # triangle pointing up: rows widen from the apex
            m = (dy >= -h) & (dy <= h) & (abs(dx) <= (dy + h) / 2)
        img[m] = COLORS[color]
    return img


def chain(rows):
    """Running sum over per-row counts: [3, 2, 4, 1] -> "3+2=5+4=9+1=10". Every step adds a number <= GRID."""
    out, total = str(rows[0]), rows[0]
    for r in rows[1:]:
        total += r
        out += f"+{r}={total}"
    return out


def parse_chain(s):
    """Inverse of chain: (rows, running sums) or None if malformed."""
    try:
        parts = s.split("+")
        rows, sums = [int(parts[0])], []
        for p in parts[1:]:
            r, t = p.split("=")
            rows.append(int(r))
            sums.append(int(t))
        return rows, sums
    except ValueError:
        return None


def row_counts(objs):
    return [sum(o[2] == r for o in objs) for r in range(GRID)]


def sum_question(rng):
    """Text-only arithmetic in the same chain format (the image is blank): teaches small sums up to GRID * GRID."""
    rows = [int(x) for x in rng.integers(0, GRID + 1, GRID)]
    return f"What is {'+'.join(map(str, rows))}?", chain(rows)


def question(rng, objs, rows=False):
    """A random question about the scene and its exact answer (the verifier is string equality).
    rows=True answers "How many shapes" by counting row by row (`chain`) instead of with a single number."""
    kind = rng.integers(5)
    shape, color = str(rng.choice(SHAPES)), str(rng.choice(list(COLORS)))
    if kind == 1:
        return f"How many {color} shapes are there?", str(sum(o[1] == color for o in objs))
    if kind == 2:
        return f"How many {shape}s are there?", str(sum(o[0] == shape for o in objs))
    if kind == 3:
        if rng.random() < 0.5:  # ask about a present object half the time, so "yes" is not rare
            shape, color = objs[rng.integers(len(objs))][:2]
        return f"Is there a {color} {shape}?", "yes" if any(o[:2] == (shape, color) for o in objs) else "no"
    if kind == 4:
        unique = [o for o in objs if sum(p[0] == o[0] for p in objs) == 1]
        if unique:
            o = unique[rng.integers(len(unique))]
            return f"What color is the {o[0]}?", o[1]
    return "How many shapes are there?", chain(row_counts(objs)) if rows else str(len(objs))


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


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def continue_story(doc):
    """A Korean story as an instruction (TinyStories-Korean has no Instruct version): the first sentence is the prompt,
    the rest of the story is the answer. Returns None for one-sentence stories."""
    parts = _SENTENCE_END.split(doc.strip(), maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        return None
    return f"다음 문장으로 시작하는 짧은 이야기를 써 줘.\n시작: {parts[0]}", parts[1].strip()


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


def addition_accuracy(model, tok, digits=range(1, WIDTH + 1), n=200, seed=0, lang="en"):
    """Greedy pass@1 for d-digit + d-digit problems, per d. Fixed width: every prompt has the same token length."""
    rng = np.random.default_rng(seed)
    acc = {}
    for d in digits:
        probs = [addition(rng, d, lang=lang)[0] for _ in range(n)]
        outs = reply(model, tok, probs, max_new=WIDTH + 2)
        acc[d] = sum(check_addition(p, o) for p, o in zip(probs, outs)) / n
    return acc

"""Tensor reverse-mode autograd on numpy, from scratch.

Every Tensor remembers its parents (`_prev`) and, per parent, a function that maps
the upstream gradient dL/d(out) to that parent's contribution dL/d(parent)
(chain rule). `backward()` applies them in reverse topological order.
This is what torch.autograd does; stage 2+ uses torch for the GPU.
"""
import numpy as np


def _unbroadcast(g, shape):
    """Sum a gradient back down to `shape`, undoing numpy broadcasting."""
    while g.ndim > len(shape):
        g = g.sum(0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(i, keepdims=True)
    return g


class Tensor:
    def __init__(self, data, _prev=(), _fns=()):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = None
        self._prev, self._fns = _prev, _fns

    @property
    def shape(self):
        return self.data.shape

    def _out(self, data, *parents_fns):
        return Tensor(data, tuple(p for p, _ in parents_fns), tuple(f for _, f in parents_fns))

    def __add__(self, o):
        o = o if isinstance(o, Tensor) else Tensor(o)
        return self._out(self.data + o.data, (self, lambda g: _unbroadcast(g, self.shape)), (o, lambda g: _unbroadcast(g, o.shape)))

    def __mul__(self, o):
        o = o if isinstance(o, Tensor) else Tensor(o)
        return self._out(
            self.data * o.data,
            (self, lambda g: _unbroadcast(g * o.data, self.shape)),
            (o, lambda g: _unbroadcast(g * self.data, o.shape)),
        )

    def __matmul__(self, o):  # (..., n, k) @ (..., k, m)
        a, b = self.data, o.data
        return self._out(
            a @ b,
            (self, lambda g: _unbroadcast(g @ b.swapaxes(-1, -2), self.shape)),
            (o, lambda g: _unbroadcast(a.swapaxes(-1, -2) @ g, o.shape)),
        )

    def __pow__(self, k):  # k is a plain number
        return self._out(self.data**k, (self, lambda g: g * k * self.data ** (k - 1)))

    def exp(self):
        e = np.exp(self.data)
        return self._out(e, (self, lambda g: g * e))

    def log(self):
        return self._out(np.log(self.data), (self, lambda g: g / self.data))

    def tanh(self):
        t = np.tanh(self.data)
        return self._out(t, (self, lambda g: g * (1 - t * t)))

    def relu(self):
        return self._out(np.maximum(self.data, 0), (self, lambda g: g * (self.data > 0)))

    def sum(self, axis=None, keepdims=False):
        def fn(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            return np.broadcast_to(g, self.shape)

        return self._out(self.data.sum(axis, keepdims=keepdims), (self, fn))

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.shape[axis]
        return self.sum(axis, keepdims) * (1 / n)

    def reshape(self, *shape):
        return self._out(self.data.reshape(*shape), (self, lambda g: g.reshape(self.shape)))

    def __getitem__(self, idx):  # E[token_ids] is an embedding lookup
        def fn(g):
            z = np.zeros_like(self.data)
            np.add.at(z, idx, g)  # repeated ids accumulate, unlike z[idx] += g
            return z

        return self._out(self.data[idx], (self, fn))

    __radd__ = __add__
    __rmul__ = __mul__

    def __neg__(self):
        return self * -1

    def __sub__(self, o):
        return self + (-o)

    def __truediv__(self, o):
        return self * (o**-1 if isinstance(o, Tensor) else 1 / o)

    def backward(self):
        order, seen = [], set()

        def visit(t):  # post-order DFS = topological order
            if id(t) not in seen:
                seen.add(id(t))
                for p in t._prev:
                    visit(p)
                order.append(t)

        visit(self)
        for t in order:
            t.grad = np.zeros_like(t.data)
        self.grad = np.ones_like(self.data)
        for t in reversed(order):
            for p, fn in zip(t._prev, t._fns):
                p.grad = p.grad + fn(t.grad)  # += : a tensor may feed several ops


def cross_entropy(logits, targets):
    """Mean softmax cross-entropy over rows of (N, C) logits; fused backward = (softmax - onehot) / N."""
    z = logits.data - logits.data.max(1, keepdims=True)  # shift for numerical stability
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    n = len(targets)

    def fn(g):
        d = np.exp(logp)
        d[np.arange(n), targets] -= 1
        return g * d / n

    return logits._out(-logp[np.arange(n), targets].mean(), (logits, fn))


if __name__ == "__main__":
    # Byte-level MLP language model (Bengio et al. 2003): previous 8 bytes -> next byte.
    import sys
    import time

    path = sys.argv[1] if len(sys.argv) > 1 else "data/TinyStoriesV2-GPT4-valid.txt"
    data = np.frombuffer(open(path, "rb").read(5_000_000), dtype=np.uint8).astype(np.int64)
    split = int(0.9 * len(data))
    rng = np.random.default_rng(0)
    ctx, emb, hid, V = 8, 24, 256, 256
    E = Tensor(rng.normal(0, 1, (V, emb)))
    W1 = Tensor(rng.normal(0, (ctx * emb) ** -0.5, (ctx * emb, hid)))
    b1 = Tensor(np.zeros(hid))
    W2 = Tensor(rng.normal(0, hid**-0.5, (hid, V)) * 0.1)  # small init -> near-uniform first predictions
    b2 = Tensor(np.zeros(V))
    params = [E, W1, b1, W2, b2]
    print(f"{sum(p.data.size for p in params):,} parameters, {split:,} train bytes")

    def batch(lo, hi, n=128):
        i = rng.integers(lo, hi - ctx, n)
        return data[i[:, None] + np.arange(ctx)], data[i + ctx]

    def forward(x):
        h = (E[x].reshape(len(x), ctx * emb) @ W1 + b1).tanh()
        return h @ W2 + b2

    steps, t0 = 6000, time.time()
    for step in range(steps + 1):
        x, y = batch(0, split)
        loss = cross_entropy(forward(x), y)
        loss.backward()
        lr = 0.1 if step < steps * 0.7 else 0.01
        for p in params:
            p.data -= lr * p.grad
        if step % 1000 == 0:
            xv, yv = batch(split, len(data), 4096)
            val = cross_entropy(forward(xv), yv).data
            print(f"step {step:5d} train {loss.data:.3f} val {val:.3f} ({time.time() - t0:.0f}s)")

    out = list(b"Once upon a time")
    for _ in range(300):
        p = np.exp(forward(np.array([out[-ctx:]])).data[0])
        out.append(rng.choice(V, p=p / p.sum()))
    print(bytes(out).decode("utf-8", errors="replace"))

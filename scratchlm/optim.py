"""AdamW, LR schedule and gradient clipping, written by hand (no torch.optim).

Adam keeps per-parameter EMAs of the gradient (m, 1st moment) and its square (v, 2nd moment):
    m = b1 m + (1-b1) g        v = b2 v + (1-b2) g^2
    p -= lr * m_hat / (sqrt(v_hat) + eps),   m_hat = m / (1-b1^t), v_hat = v / (1-b2^t)   (bias correction)
so each coordinate moves by ~lr regardless of its gradient scale. AdamW applies weight decay directly to
the weights (p -= lr * wd * p) instead of adding wd*p to g, so decay is not rescaled by 1/sqrt(v).
"""
import math

import torch


class AdamW:
    def __init__(self, params, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        self.params = [p for p in params if p.requires_grad]
        self.betas, self.eps, self.wd = betas, eps, weight_decay
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    @torch.no_grad()
    def step(self, lr):
        self.t += 1
        b1, b2 = self.betas
        c1, c2 = 1 - b1**self.t, 1 - b2**self.t
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            if p.ndim >= 2:  # decay matrices only; gains (RMSNorm weights) stay undecayed
                p.mul_(1 - lr * self.wd)
            m.lerp_(p.grad, 1 - b1)
            v.lerp_(p.grad.square(), 1 - b2)
            p.addcdiv_(m / c1, (v / c2).sqrt_().add_(self.eps), value=-lr)

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def state_dict(self):
        return {"t": self.t, "m": self.m, "v": self.v}

    def load_state_dict(self, s):
        self.t = s["t"]
        for dst, src in zip(self.m + self.v, s["m"] + s["v"]):
            dst.copy_(src)


def cosine_lr(step, max_lr, warmup, total, min_ratio=0.1):
    """Linear warmup to max_lr, then cosine decay to min_ratio * max_lr at `total`."""
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return max_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def clip_grad_norm(params, max_norm):
    """Scale all grads so their global L2 norm is <= max_norm. Returns the norm before clipping."""
    grads = [p.grad for p in params if p.grad is not None]
    norm = torch.sqrt(sum(g.float().square().sum() for g in grads))
    scale = (max_norm / (norm + 1e-6)).clamp(max=1.0)  # tensor math: no GPU->CPU sync
    for g in grads:
        g.mul_(scale)
    return norm

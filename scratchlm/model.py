"""Decoder-only Transformer (GPT), every layer written by hand on torch tensors.

ids -> Embedding -> N x Block -> RMSNorm -> logits (tied to the embedding by default)
Block (pre-norm residual):  x = x + Attn(RMSNorm(x));  x = x + SwiGLU(RMSNorm(x))
torch supplies tensors, autograd and the GPU; scratchlm/autograd.py shows what autograd does.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class Config:
    vocab_size: int = 8192
    ctx_len: int = 512
    n_layer: int = 8
    n_head: int = 8
    d_model: int = 512
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    sdpa: bool = True  # False = hand-written attention (same math, materializes the T x S matrix)
    causal: bool = True  # False = every token sees every token (the vision encoder)

    @property
    def d_ff(self):  # SwiGLU has 3 matrices, so 8/3*d keeps params equal to a 4*d two-matrix MLP
        return 64 * math.ceil(8 * self.d_model / 3 / 64)


class Linear(nn.Module):
    def __init__(self, d_in, d_out, std=0.02):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_out, d_in) * std)

    def forward(self, x):
        return x @ self.weight.T


class Embedding(nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n, d) * 0.02)

    def forward(self, ids):
        return self.weight[ids]  # a row lookup; backward scatters-adds into those rows


class RMSNorm(nn.Module):
    """x / sqrt(mean(x^2) + eps) * g. LayerNorm without mean-centering or bias."""

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        xf = x.float()  # normalize in fp32 even under bf16 autocast
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


def rope_tables(head_dim, ctx_len, theta):
    """Angle pos * theta^(-2i/d) for each position and frequency pair i."""
    inv_freq = theta ** -(torch.arange(0, head_dim, 2).float() / head_dim)
    ang = torch.outer(torch.arange(ctx_len).float(), inv_freq)  # (T, head_dim/2)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    """Rotate each pair (x1_i, x2_i), taken from the two halves of the head dim, by its angle."""
    x1, x2 = x.float().chunk(2, -1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).type_as(x)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n_head, self.head_dim, self.sdpa, self.causal = c.n_head, c.d_model // c.n_head, c.sdpa, c.causal
        self.qkv = Linear(c.d_model, 3 * c.d_model)
        self.proj = Linear(c.d_model, c.d_model, std=0.02 / math.sqrt(2 * c.n_layer))

    def forward(self, x, cos, sin, cache=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)  # each (B, H, T, hd)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:  # KV cache: keep past keys/values, compute only the new queries
            if cache:
                k, v = torch.cat([cache["k"], k], 2), torch.cat([cache["v"], v], 2)
            cache["k"], cache["v"] = k, v
        S = k.shape[2]
        mask = torch.ones(T, S, dtype=torch.bool, device=x.device)
        if self.causal:
            mask = mask.tril(S - T)  # query t sees keys <= its position
        if self.sdpa:
            if not self.causal:
                y = F.scaled_dot_product_attention(q, k, v)
            else:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True) if T == S else F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            att = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            y = att.masked_fill(~mask, float("-inf")).softmax(-1) @ v
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


def silu(x):
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate, self.up = Linear(c.d_model, c.d_ff), Linear(c.d_model, c.d_ff)
        self.down = Linear(c.d_ff, c.d_model, std=0.02 / math.sqrt(2 * c.n_layer))

    def forward(self, x):
        return self.down(silu(self.gate(x)) * self.up(x))  # silu(gate) acts as a learned, smooth gate on up


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm1, self.attn, self.norm2, self.mlp = RMSNorm(c.d_model), Attention(c), RMSNorm(c.d_model), SwiGLU(c)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.norm1(x), cos, sin, cache)
        return x + self.mlp(self.norm2(x))


def cross_entropy(logits, targets):
    """Mean of logsumexp(z) - z[target] over positions with target >= 0 (-1 = ignore, used for SFT masking)."""
    z, t = logits.float().flatten(0, -2), targets.flatten()
    keep = t >= 0
    nll = z.logsumexp(-1) - z.gather(1, t.clamp(min=0)[:, None]).squeeze(1)
    return (nll * keep).sum() / keep.sum().clamp(min=1)  # all-ignored batch -> 0, not NaN


class GPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.embed = Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.norm = RMSNorm(c.d_model)
        self.head = None if c.tie_embeddings else Linear(c.d_model, c.vocab_size)
        cos, sin = rope_tables(c.d_model // c.n_head, c.ctx_len, c.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, ids, targets=None, caches=None, embeds=None):
        """`embeds` replaces the token embeddings (the VLM splices image features into them); `ids` still gives the length."""
        T = ids.shape[1]
        pos = caches[0]["k"].shape[2] if caches and caches[0] else 0
        assert pos + T <= self.c.ctx_len, "sequence longer than ctx_len"
        cos, sin = self.cos[pos : pos + T], self.sin[pos : pos + T]
        x = self.embed(ids) if embeds is None else embeds
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, caches[i] if caches else None)
        logits = self.norm(x) @ (self.embed if self.head is None else self.head).weight.T
        return logits if targets is None else (logits, cross_entropy(logits, targets))

    @torch.no_grad()
    def generate(self, ids, max_new, temperature=1.0, top_k=None, embeds=None):
        caches = [{} for _ in self.blocks]
        logits = self(ids, caches=caches, embeds=embeds)[:, -1]  # prefill: whole prompt in one pass
        for _ in range(max_new):
            if temperature == 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits.float() / temperature
                if top_k:
                    logits[logits < logits.topk(top_k).values[:, -1:]] = float("-inf")
                nxt = torch.multinomial(logits.softmax(-1), 1)
            ids = torch.cat([ids, nxt], 1)
            if ids.shape[1] >= self.c.ctx_len:
                break
            logits = self(nxt, caches=caches)[:, -1]  # decode: one new token, past K/V from the cache
        return ids


def load_checkpoint(path, device="cuda", vocab_size=None):
    """Rebuild a GPT from a train.py checkpoint. A larger vocab_size appends embedding rows for new tokens
    (vocab surgery): mean of the existing rows plus small noise, so they start as an "average" token."""
    ck = torch.load(path, map_location=device)
    c, state = Config(**ck["model_config"]), ck["model"]
    if vocab_size and vocab_size > c.vocab_size:
        for k in ("embed.weight", "head.weight"):
            if k in state:
                w = state[k]
                new = w.mean(0, keepdim=True) + 0.02 * torch.randn(vocab_size - c.vocab_size, w.shape[1], device=w.device)
                state[k] = torch.cat([w, new])
        c.vocab_size = vocab_size
    model = GPT(c).to(device)
    model.load_state_dict(state)
    return model


if __name__ == "__main__":
    # Parameter breakdown + training-step throughput: hand-written attention vs SDPA, fp32 vs bf16.
    import time

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    for name, c in {"small": Config(), "base": Config(n_layer=12, n_head=12, d_model=768)}.items():
        m = GPT(c)
        emb = m.embed.weight.numel()
        print(f"[{name}] params {sum(p.numel() for p in m.parameters()) / 1e6:.1f}M (embedding {emb / 1e6:.1f}M), d_ff {c.d_ff}")
    c = Config()
    B, T = 16, c.ctx_len
    ids = torch.randint(0, c.vocab_size, (B, T + 1), device=dev)
    for sdpa in (False, True):
        for dtype in (torch.float32, torch.bfloat16):
            torch.manual_seed(0)
            m = GPT(Config(sdpa=sdpa)).to(dev)
            torch.cuda.reset_peak_memory_stats() if dev == "cuda" else None
            for i in range(13):  # 3 warmup + 10 timed fwd+bwd steps
                if i == 3:
                    torch.cuda.synchronize() if dev == "cuda" else None
                    t0 = time.time()
                with torch.autocast(dev, dtype=dtype, enabled=dtype != torch.float32):
                    _, loss = m(ids[:, :-1], ids[:, 1:])
                loss.backward()
                m.zero_grad(set_to_none=True)
            torch.cuda.synchronize() if dev == "cuda" else None
            tps = 10 * B * T / (time.time() - t0)
            mem = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0
            print(f"attn={'sdpa' if sdpa else 'manual'} {str(dtype)[6:]:8s} {tps / 1e3:6.1f}k tok/s  peak {mem:.1f} GiB  loss {loss.item():.3f}")
    print(f"initial loss should be ~ln(V) = {math.log(c.vocab_size):.3f}")

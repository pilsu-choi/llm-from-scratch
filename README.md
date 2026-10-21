# llm-from-scratch

A learning project: build an LLM, then a VLM, **from scratch** — no pretrained weights, and dependencies are just `numpy` and `torch` (torch for GPU tensors + autograd only; layers and optimizers are hand-written),
then make it improve itself with a **recursive self-improvement (RSI)** loop on verifiable tasks.

| Stage | What | Status |
|---|---|---|
| 0 | PGX (GB10) + NGC PyTorch container | ✅ |
| 1 | Byte-level BPE tokenizer (`scratchlm/tokenizer.py`, stdlib only) | ✅ |
| 1.5 | Tensor autograd on numpy + byte-level MLP LM (`scratchlm/autograd.py`) | ✅ |
| 2 | Decoder-only GPT, hand-written layers (RMSNorm, RoPE, SwiGLU, KV cache) — `scratchlm/model.py` | ✅ |
| 3 | Pretraining on TinyStories | ⏳ |
| 4 | SFT | ⏳ |
| 5 | RSI: generate → verify → retrain (STaR / ReST-EM) | ⏳ |
| 6 | VLM: ViT + projector (LLaVA-style) | ⏳ |
| 7 | Korean extension | ⏳ |

Reference implementations for a later comparison: Karpathy's microgpt → nanoGPT → nanochat.

Agent rules: [AGENTS.md](AGENTS.md). Goals and done-when per stage: [PRD.md](PRD.md).

Details and principles (Korean): [wiki/](wiki/index.md), starting with the [roadmap](wiki/2026-10-21-roadmap.md).

## Quickstart

```bash
uv run --no-project --with numpy python -m unittest -v                  # local tests (stdlib unittest)
ssh pgx 'cd llm-from-scratch && bash scripts/get_data.sh'          # TinyStories on PGX
scripts/pgx.sh python -m unittest -v                               # all tests incl. torch, on the GB10
scripts/pgx.sh python -m scratchlm.model                           # param counts + throughput benchmark
scripts/pgx.sh python -m scratchlm.tokenizer data/TinyStoriesV2-GPT4-train.txt data/tok8k.json --vocab 8192
```

`scripts/pgx.sh` rsyncs the repo to PGX and runs the command in `nvcr.io/nvidia/pytorch:26.09-py3`.

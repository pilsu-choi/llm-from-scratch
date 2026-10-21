# llm-from-scratch

A learning project: build an LLM, then a VLM, **from scratch** (no pretrained weights, no HF model classes),
then make it improve itself with a **recursive self-improvement (RSI)** loop on verifiable tasks.

| Stage | What | Status |
|---|---|---|
| 0 | PGX (GB10) + NGC PyTorch container | ✅ |
| 1 | Byte-level BPE tokenizer (`scratchlm/tokenizer.py`, stdlib only) | ✅ |
| 2 | Decoder-only GPT (RMSNorm, RoPE, SwiGLU, KV cache) | ⏳ |
| 3 | Pretraining on TinyStories | ⏳ |
| 4 | SFT | ⏳ |
| 5 | RSI: generate → verify → retrain (STaR / ReST-EM) | ⏳ |
| 6 | VLM: ViT + projector (LLaVA-style) | ⏳ |
| 7 | Korean extension | ⏳ |

Details and principles (Korean): [wiki/](wiki/index.md), starting with the [roadmap](wiki/2026-10-21-roadmap.md).

## Quickstart

```bash
uv run --no-project --with pytest pytest -q tests                  # local tests
ssh pgx 'cd llm-from-scratch && bash scripts/get_data.sh'          # TinyStories on PGX
scripts/pgx.sh python -m scratchlm.tokenizer data/TinyStoriesV2-GPT4-train.txt data/tok8k.json --vocab 8192
```

`scripts/pgx.sh` rsyncs the repo to PGX and runs the command in `nvcr.io/nvidia/pytorch:26.09-py3`.

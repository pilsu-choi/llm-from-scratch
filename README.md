# llm-from-scratch

A learning project: build an LLM, then a VLM, **from scratch** — no pretrained weights, and dependencies are just `numpy` and `torch` (torch for GPU tensors + autograd only; layers and optimizers are hand-written),
then make it improve itself with a **recursive self-improvement (RSI)** loop on verifiable tasks.

| Stage | What | Status |
|---|---|---|
| 0 | PGX (GB10) + NGC PyTorch container | ✅ |
| 1 | Byte-level BPE tokenizer (`scratchlm/tokenizer.py`, stdlib only) | ✅ |
| 1.5 | Tensor autograd on numpy + byte-level MLP LM (`scratchlm/autograd.py`) | ✅ |
| 2 | Decoder-only GPT, hand-written layers (RMSNorm, RoPE, SwiGLU, KV cache) — `scratchlm/model.py` | ✅ |
| 3 | Pretraining on TinyStories: hand-written AdamW, cosine LR, bf16, torch.compile — `scratchlm/train.py` | ✅ |
| 4 | SFT: chat template, role tokens (vocab surgery), prompt masking, addition + story instructions — `scratchlm/sft.py`, `tasks.py` | ✅ |
| 5 | RSI: generate → verify → retrain on own answers — `scratchlm/rsi.py`. 16-digit addition 7% → 100% in 10 rounds (41 min) | ✅ |
| 6 | VLM: ViT + projector (LLaVA-style) — `scratchlm/vision.py`. Stage A (LM frozen, 17 min): 57.7% on synthetic-scene QA, 1 object 97%, unseen 7–10 objects 39→19% | 🔄 |
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
scripts/pgx.sh python -m scratchlm.data data/TinyStoriesV2-GPT4-train.txt data/tok8k.json data/ts_train.bin   # (and valid -> ts_val.bin)
DETACH=pretrain-small scripts/pgx.sh python -m scratchlm.train --name small --steps 16000   # ~1 h; rerun to resume
scripts/pgx.sh python -m scratchlm.sft prepare      # SFT shards (needs TinyStories-Instruct in data/)
scripts/pgx.sh python -m scratchlm.train --name sft --init runs/small/ckpt.pt --tok data/tok_chat.json --train data/sft_train.bin --val data/sft_val.bin --steps 2600 --lr 3e-4 --warmup 100   # (stage 5 used --name sft_pad)
scripts/pgx.sh python -m scratchlm.sft eval runs/sft/ckpt.pt
DETACH=rsi scripts/pgx.sh python -m scratchlm.rsi --init runs/sft_pad/ckpt.pt --rounds 10   # ~4 min/round
scripts/pgx.sh rm -rf runs/<name>   # runs/ is root-owned: delete from inside the container
```

`scripts/pgx.sh` rsyncs the repo to PGX and runs the command in `nvcr.io/nvidia/pytorch:26.09-py3`.

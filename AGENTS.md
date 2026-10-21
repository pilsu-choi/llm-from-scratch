# AGENTS.md — llm-from-scratch

Instructions for coding agents (Claude Code, etc.) working in this repository.
Read this whole file and [PRD.md](PRD.md) before writing code. When this file and a request conflict, ask.

## 1. What we are building

A learning project: an LLM, then a VLM, built **from scratch**, finished with a **recursive self-improvement (RSI)** loop
in which the model generates answers, a program verifier keeps the correct ones, and the model retrains on them.
Goals, scope and the done-when criteria for each stage are in [PRD.md](PRD.md). The step-by-step plan is in
[wiki/2026-10-21-roadmap.md](wiki/2026-10-21-roadmap.md).

## 2. The owner is learning

- Understanding comes before features. Each stage ships **code + a Korean concept page + measured results**.
- Write plain, readable code that maps to the math. Comment the *why* (a derivation, a shape, a numerical trick), not the what.
- Work **one stage at a time**. Finish a stage, report it, and wait for the owner's go-ahead before starting the next.
- When a stage is done, summarize in 3–5 bullets what it teaches, and lead with the measured numbers.

## 3. Hard rules (never break these)

1. **No pretrained weights**, ever. No Hugging Face `transformers`/`tokenizers`/`datasets`.
2. **Dependencies: `numpy` and `torch` only** (plus stdlib). Ask before adding any other package.
   - From torch, use only tensors, autograd, `nn.Module`/`nn.Parameter`, and `F.scaled_dot_product_attention`
     (only after a hand-written version exists and a test shows they are equal).
   - Do not use `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `nn.Transformer*` or `torch.optim`. Write the layers,
     the optimizer (AdamW) and the LR schedule yourself. `torch.nn.functional` references are allowed **in tests only**.
   - The tokenizer, data prep, RSI verifiers and synthetic images use stdlib/numpy only. Downloads use `curl`.
3. **Verify every hand-written op.** Use finite-difference gradchecks for the numpy autograd, and equivalence tests
   for torch code (hand-written vs reference, KV cache vs full forward, causality).
4. **Only measured numbers.** Wiki and README numbers come from a run you did. Label an estimate as one.
5. **Don't copy reference code.** Karpathy's microgpt → nanoGPT → nanochat are named for a *later comparison*.
   Keep the design choices comparable and record each one in the concept pages, but write our own code.
6. **PGX is shared.** The vLLM containers `embed` and `rerank` (rag-bench) use about 30% of GPU memory. Never stop them without asking.
   `data/` and `runs/` live on PGX only and never go into git.

## 4. Repository layout

```
scratchlm/         the package: tokenizer.py, autograd.py (numpy), model.py, data.py, optim.py, train.py (torch), tasks.py (chat template + verifiers), sft.py, rsi.py, vision.py (ViT + VLM)
tests/             stdlib unittest; torch tests skip when torch is not installed
scripts/pgx.sh     rsync the repo to PGX, run a command in the NGC PyTorch container
scripts/get_data.sh
wiki/              OKF knowledge bundle (Korean): roadmap, concepts/, runbooks/, decisions/, log.md
AGENTS.md PRD.md README.md
```

## 5. Commands

```bash
uv run --no-project --with numpy python -m unittest -v   # local: tokenizer + autograd (torch tests skip)
scripts/pgx.sh python -m unittest -v                      # all tests on the GB10
scripts/pgx.sh python -m scratchlm.<module>               # each module's __main__ is its demo/benchmark
DETACH=<name> scripts/pgx.sh python -m scratchlm.train --name <run>   # long run in a detached container; rerun resumes
```

Before declaring a task done, all tests must pass on PGX.
`scripts/pgx.sh` syncs **the directory it is run from**, so finish by running the tests again from the main checkout after a merge.

## 6. Code conventions

- **Concise: one concept, one implementation.** No two functions or classes with overlapping meaning. Reuse
  (`cross_entropy`, `Linear`, ...) instead of re-creating. Each module's `__main__` is its demo, not a separate script.
- The numpy autograd is float64, for gradcheck. Torch training uses bf16 autocast. NGC forces TF32 for fp32 matmuls by default,
  and `pgx.sh` turns that off (`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0`).
- Targets of `-1` mean "ignore" in losses (used for SFT prompt masking).
- Commit messages: `<stage>: <what>` (e.g. `s3: add AdamW with decoupled weight decay`), ending with the attribution lines.

## 7. Workflow

- `dev` is the default branch (GitHub `pilsu-choi/llm-from-scratch`, public).
- For each task, create a **git worktree on a new branch from `dev`** (`git worktree add ../llm-from-scratch-<task> -b <task> dev`).
  Commit there, merge into `dev`, push `dev`, then remove the worktree.
- Refresh `README.md` (stage table, commands) before every push.
- Split independent work (for example a concept page alongside code) across parallel sub-agents. Integration, review and commits stay with the orchestrator.
  Review every sub-agent page against the code and the measured numbers before committing it.

## 8. Wiki (record every piece of work)

`wiki/` is an Open Knowledge Format (OKF v0.1) bundle, written in Korean.

- Add a dated entry (UTC) to `wiki/log.md` for every change.
- New files get a date prefix: `concepts/YYYY-MM-DD-<slug>.md`. `index.md` and `log.md` keep their names.
- Every page starts with YAML frontmatter: `type` (required), `title`, `description`, `tags`, `timestamp`,
  and `resource` (code path) for pages about code. Every directory has an `index.md`. Link pages with relative links.
- One concept page per stage: principles, derivations, what the tests guarantee, measured results, exercises, and a
  "comparison with the reference implementations" table (filled in later).

## 9. When unsure

Stop and ask the owner about new dependencies, anything touching the hard rules, scope changes, and before moving to the next stage.

## 10. Owner decisions (2026-10-21)

- RSI means Recursive Self-Improvement with a program verifier (STaR / ReST-EM). A small model does not judge its own outputs.
- Compute is PGX (GB10, aarch64, CUDA 13) through the NGC `pytorch:26.09-py3` container. The local RTX 4070 is for debugging.
- English (TinyStories) first, then extend to Korean.
- Minimal dependencies: first torch only, then numpy allowed too, for speed. Stage 1.5 is a numpy tensor autograd.
- Digits are tokenized one per token, for the arithmetic RSI task.
- Addition format (stage 5): operands zero-padded to 16 digits, answer reversed and padded to 17, half the SFT problems shifted to random positions;
  SFT trains spans of 1–7 digits. Plain and reversed-only formats failed to length-generalize (wiki/lessons).
- The repo is public on GitHub.

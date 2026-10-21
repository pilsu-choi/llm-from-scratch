#!/usr/bin/env bash
# Sync the repo to PGX and run a command inside the NGC PyTorch container.
#   scripts/pgx.sh python -m scratchlm.tokenizer data/...   (cwd = repo root)
# data/ and runs/ live only on PGX (excluded from sync, never deleted by it).
set -euo pipefail
HOST=${PGX_HOST_SSH:-pgx}
IMAGE=${IMAGE:-nvcr.io/nvidia/pytorch:26.09-py3}
REMOTE=llm-from-scratch
cd "$(dirname "$0")/.."
rsync -az --delete --exclude data/ --exclude runs/ --exclude .git/ --exclude __pycache__/ ./ "$HOST:$REMOTE/"
ssh -t "$HOST" "mkdir -p $REMOTE/data $REMOTE/runs && docker run --rm --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v \$HOME/$REMOTE:/ws -w /ws -e PYTHONPATH=/ws $IMAGE $(printf '%q ' "$@")"

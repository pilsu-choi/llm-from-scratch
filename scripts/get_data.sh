#!/usr/bin/env bash
# Download TinyStories (GPT-4 generated, English) into data/. Run on PGX: scripts/pgx.sh bash scripts/get_data.sh
set -euo pipefail
mkdir -p data && cd data
base=https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main
for f in TinyStoriesV2-GPT4-valid.txt TinyStoriesV2-GPT4-train.txt; do
  [ -s "$f" ] || curl -fL --retry 3 -o "$f" "$base/$f"
done
ls -lh

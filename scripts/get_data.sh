#!/usr/bin/env bash
# Download TinyStories (GPT-4 generated, English) into data/. Run on PGX: scripts/pgx.sh bash scripts/get_data.sh
# `scripts/get_data.sh ko` downloads the Korean translation (g0ster/TinyStories-Korean, MIT, machine-translated with EEVE) for stage 7.
set -euo pipefail
mkdir -p data && cd data
if [ "${1:-}" = ko ]; then
  base=https://huggingface.co/datasets/g0ster/TinyStories-Korean/resolve/main
  files="TinyStories-Korean-valid.txt TinyStories-Korean-train.txt"
else
  base=https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main
  files="TinyStoriesV2-GPT4-valid.txt TinyStoriesV2-GPT4-train.txt"
fi
for f in $files; do
  [ -s "$f" ] || curl -fL --retry 3 -o "$f" "$base/$f"
done
ls -lh

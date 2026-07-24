#!/usr/bin/env bash
# Exp1 — stage all renders (existing scenes + resamples), embed on kiti GPU 0,
# pull back embeddings.npz.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$ROOT/exp1_geometry/all_images"

rm -rf "$STAGE"; mkdir -p "$STAGE"
for d in "$ROOT"/motifs/*/; do
  m=$(basename "$d")
  mkdir -p "$STAGE/$m"
  cp "$d"scene_*.png "$STAGE/$m/" 2>/dev/null || true
done
for d in "$ROOT"/exp1_geometry/renders/*/; do
  m=$(basename "$d")
  mkdir -p "$STAGE/$m"
  cp "$d"resample_*.png "$STAGE/$m/" 2>/dev/null || true
done
echo "staged: $(find "$STAGE" -name '*.png' | wc -l) images"

ssh kiti 'mkdir -p ~/exp1_geb'
rsync -a --delete "$STAGE/" kiti:~/exp1_geb/images/
rsync -a "$ROOT/exp1_geometry/embed_remote.py" kiti:~/exp1_geb/
ssh kiti 'cd ~/exp1_geb && http_proxy=http://100.73.92.22:7890 https_proxy=http://100.73.92.22:7890 CUDA_VISIBLE_DEVICES=0 ~/micromamba/envs/interp/bin/python embed_remote.py images embeddings.npz'
rsync -a kiti:~/exp1_geb/embeddings.npz "$ROOT/exp1_geometry/embeddings.npz"
echo done

#!/usr/bin/env bash
# GEB v2 full re-eval, frontier models via yunwu :stable.
# Run AFTER: stories/scenes complete -> verify_scenes_v2.py -> build_items_v2.py
# Usage: bash eval/run_frontier_v2.sh [--smoke]
set -u
cd "$(dirname "$0")/.."

LIMIT=""
if [ "${1:-}" = "--smoke" ]; then LIMIT="--limit 30"; fi

for spec in "gpt-5.5:stable api_gpt-5.5" \
            "gpt-5.2:stable api_gpt-5.2" \
            "claude-opus-4-8:stable api_claude-opus-4-8"; do
  model="${spec%% *}"; tag="${spec##* }"
  echo "=== $model -> $tag ==="
  PYTHONUNBUFFERED=1 python3 eval/run_eval_v2.py \
    --base "$OPENAI_BASE_URL" --model "$model" --tag "$tag" \
    --vision --adv --workers 8 $LIMIT \
    2>&1 | tee "eval/results_v2/${tag}.log"
done
echo "FRONTIER_V2_DONE"

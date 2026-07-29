#!/bin/bash
set -u
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
for mt in "gpt-5.5:stable|yw_gpt-5.5" \
          "gpt-5.2:stable|yw_gpt-5.2" \
          "claude-opus-4-8:stable|yw_claude-opus-4-8"; do
  m="${mt%%|*}"; t="${mt##*|}"
  if [ -s "catlab/results/$t.json" ]; then echo "=== $t: exists, skip ==="; continue; fi
  echo "=== catlab $m -> $t ==="
  python3 catlab/run_catlab.py --base "$OPENAI_BASE_URL" --key "$OPENAI_API_KEY" \
    --model "$m" --workers 6 --tag "$t" 2>&1 | tail -40
done
echo "CATLAB_YW_ALL_DONE"

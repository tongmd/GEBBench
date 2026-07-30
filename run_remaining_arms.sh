#!/usr/bin/env bash
# Serial driver for the remaining GEB-Bench arms.
#
# All frontier endpoints are geoblocked on direct egress (403 "not available in
# your region"), so every call goes through the single SOCKS tunnel on :1080.
# That tunnel drops connections once more than ~8-10 long requests share it,
# and a dropped call lands as an empty reply that scores as wrong -- an
# accuracy of 0.0000 that looks like a real result. Hence: one arm at a time,
# never in parallel, and audit every file for empty replies before it counts.
#
# The API key is read from the environment, never passed on a command line
# (argv is world-readable via ps).
set -uo pipefail
cd "$(dirname "$0")"

export GEB_PROXY="${GEB_PROXY:-socks5://127.0.0.1:1080}"
: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY must be set}"
BASE="https://openrouter.ai/api/v1"
LOG="logs/remaining_arms.log"
mkdir -p logs exp2_comparison/results

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# Wait for any still-running catlab worker to finish before adding tunnel load.
while pgrep -f "[r]un_catlab.py" >/dev/null 2>&1; do
  say "catlab still running; waiting 120s"
  sleep 120
done
say "catlab clear -- starting remaining arms"

# ---- arm 1: bilingual EN half (runner skips files that already exist) ----
say "=== bilingual EN ==="
python3 eval/analysis_out/run_bilingual_v3.py >> "$LOG" 2>&1
say "bilingual exit=$?"

# ---- arm 2: comparison, the three models with no result file ----
say "=== comparison (missing models) ==="
# Slugs verified against the live API before launch: the plain
# google/gemini-3.1-pro is rejected as invalid, only the -preview form works.
for spec in "anthropic/claude-fable-5.8:or_claude-fable-5.8" \
            "google/gemini-3.1-pro-preview:or_gemini-3.1-pro" \
            "moonshotai/kimi-k2.5:or_kimi-k2.5"; do
  slug="${spec%%:*}"; tag="${spec##*:}"
  if [ -f "exp2_comparison/results/${tag}.json" ]; then
    say "  $tag exists, skip"; continue
  fi
  say "  running $slug -> $tag"
  python3 exp2_comparison/run_comparison.py \
    --base "$BASE" --model "$slug" --tag "$tag" >> "$LOG" 2>&1
  say "  $tag exit=$?"
done

# ---- integrity audit: empty raw means the call failed, not that the model erred ----
say "=== empty-reply audit ==="
python3 - >> "$LOG" 2>&1 << 'PYEOF'
import glob, json, os
bad = []
for p in sorted(glob.glob("eval/analysis_out/bilingual_v3_*.json")
                + glob.glob("exp2_comparison/results/*.json")
                + glob.glob("catlab/results/or_*.json")):
    recs = json.load(open(p)).get("records", [])
    empty = sum(1 for r in recs if not (str(r.get("raw") or "")).strip())
    print(f"{os.path.basename(p):46s} n={len(recs):4d} empty={empty}")
    if empty:
        bad.append((p, empty))
print("CLEAN" if not bad else f"NEEDS RERUN: {bad}")
PYEOF
tail -20 "$LOG"
say "ALL REMAINING ARMS DONE"

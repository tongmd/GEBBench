#!/usr/bin/env bash
# GEB v2 re-eval, open models served on kiti.
# Phase V: vision models -- servers already live on kiti (8002=qwen2-vl-7b,
#          8003=qwen2.5-vl-32b); reach them through ssh -L tunnels.
# Phase T: text models -- serve on free GPUs one at a time, eval, kill.
#          coder-32b runs LAST: it needs GPUs 4-7, so we borrow them only
#          after the 32B vision model's eval is finished (kill its server).
# Usage: bash eval/run_open_v2.sh [--smoke]   (run after build_items_v2.py)
set -u
cd "$(dirname "$0")/.."

LIMIT=""
if [ "${1:-}" = "--smoke" ]; then LIMIT="--limit 30"; fi

tunnel() {  # tunnel <local> <remote>
  ssh -O check kiti 2>/dev/null || true
  ssh -f -N -L "$1:localhost:$2" kiti
}

# ---------- Phase V: vision ----------
tunnel 18002 8002
tunnel 18003 8003
for spec in "qwen2-vl-7b 18002" "qwen2.5-vl-32b 18003"; do
  m="${spec%% *}"; p="${spec##* }"
  echo "=== vision: $m via localhost:$p ==="
  PYTHONUNBUFFERED=1 python3 eval/run_eval_v2.py \
    --base "http://localhost:$p/v1" --model "$m" --key EMPTY \
    --vision --adv --workers 8 $LIMIT 2>&1 | tee "eval/results_v2/$m.log"
done

# ---------- Phase T: text ----------
# <name> <hf id> <gpus> <port>
TEXT_MODELS=(
  "qwen2.5-7b Qwen/Qwen2.5-7B-Instruct 0 8010"
  "qwen2.5-14b Qwen/Qwen2.5-14B-Instruct 0,1 8011"
  "deepseek-coder-v2-lite deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct 0,1 8012"
  "codegeex4-9b THUDM/codegeex4-all-9b 3 8013"
  "yi-coder-9b 01-ai/Yi-Coder-9B-Chat 3 8014"
  "qwen2.5-coder-32b Qwen/Qwen2.5-Coder-32B-Instruct 4,5,6,7 8015"
)
for spec in "${TEXT_MODELS[@]}"; do
  read -r name hf gpus port <<< "$spec"
  if [ "$name" = "qwen2.5-coder-32b" ]; then
    echo "=== freeing GPUs 4-7 (kill qwen2.5-vl-32b server) ==="
    ssh kiti 'tmux kill-session -t serve32b 2>/dev/null; pkill -f "served-model-name qwen2.5-vl-32b"; sleep 5' || true
  fi
  ntp=$(( $(echo "$gpus" | tr -cd , | wc -c) + 1 ))
  echo "=== text: $name on kiti GPU $gpus port $port (tp=$ntp) ==="
  ssh kiti "tmux kill-session -t eval_$name 2>/dev/null; tmux new-session -d -s eval_$name \
    'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; \
     CUDA_VISIBLE_DEVICES=$gpus ~/micromamba/envs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
     --model $hf --served-model-name $name --tensor-parallel-size $ntp \
     --max-model-len 8192 --gpu-memory-utilization 0.92 --port $port \
     > ~/serve_eval_$name.log 2>&1'"
  # wait for the server (up to 15 min: big models load slowly)
  up=0
  for i in $(seq 1 90); do
    if ssh kiti "curl -s -m 5 localhost:$port/v1/models" 2>/dev/null | grep -q "$name"; then up=1; break; fi
    sleep 10
  done
  if [ "$up" != 1 ]; then echo "SERVER_FAIL $name"; ssh kiti "tail -20 ~/serve_eval_$name.log"; continue; fi
  tunnel "1$port" "$port"
  PYTHONUNBUFFERED=1 python3 eval/run_eval_v2.py \
    --base "http://localhost:1$port/v1" --model "$name" --key EMPTY \
    --workers 8 $LIMIT 2>&1 | tee "eval/results_v2/$name.log"
  ssh kiti "tmux kill-session -t eval_$name 2>/dev/null"
done
echo "OPEN_V2_DONE"

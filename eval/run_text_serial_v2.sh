#!/usr/bin/env bash
# GEB v2 text-model eval: download -> serve -> eval -> delete, one at a time.
# Disk on kiti is nearly full (~59G free), so weights are evicted after use.
# Runs FROM the laptop; serving happens on kiti; eval traffic via ssh -L.
# Usage: bash eval/run_text_serial_v2.sh
set -u
cd "$(dirname "$0")/.."

# name | hf id | shards | gpus | port | keep_after (1 = don't delete)
MODELS=(
  "qwen2.5-7b Qwen/Qwen2.5-7B-Instruct 4 0 8010 0"
  "qwen2.5-14b Qwen/Qwen2.5-14B-Instruct 8 0,1 8011 0"
  "deepseek-coder-v2-lite deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct 4 0,1 8012 0"
  "codegeex4-9b THUDM/codegeex4-all-9b 4 3 8013 0"
  "yi-coder-9b 01-ai/Yi-Coder-9B-Chat 4 3 8014 0"
)

tunnel() { ssh -f -N -L "$1:localhost:$2" kiti 2>/dev/null || true; }

for spec in "${MODELS[@]}"; do
  read -r name hf shards gpus port keep <<< "$spec"
  tag_done="eval/results_v2/${name}.json"
  if [ -s "$tag_done" ]; then echo "=== $name: already done, skip ==="; continue; fi
  repo_dir="~/.cache/huggingface/hub/models--$(echo "$hf" | sed 's|/|--|')"

  echo "=== $name: ensure weights ($shards shards) ==="
  ssh kiti "n=\$(ls $repo_dir/snapshots/*/*.safetensors 2>/dev/null | wc -l); echo have \$n/$shards"
  have=$(ssh kiti "ls $repo_dir/snapshots/*/*.safetensors 2>/dev/null | wc -l" | tr -d ' ')
  if [ "${have:-0}" -lt "$shards" ]; then
    echo "  downloading $hf ..."
    ssh kiti "export https_proxy=http://127.0.0.1:8234 http_proxy=http://127.0.0.1:8234 HF_HUB_ENABLE_HF_TRANSFER=0; \
      while true; do ~/micromamba/envs/vllm/bin/huggingface-cli download $hf > ~/dl_${name}.log 2>&1; \
      n=\$(ls $repo_dir/snapshots/*/*.safetensors 2>/dev/null | wc -l); \
      inc=\$(ls $repo_dir/blobs/*.incomplete 2>/dev/null | wc -l); \
      [ \"\$n\" -ge $shards ] && [ \"\$inc\" -eq 0 ] && break; sleep 15; done; echo DL_OK"
    [ $? -ne 0 ] && { echo "DL_FAIL $name"; continue; }
  fi

  ntp=$(( $(echo "$gpus" | tr -cd , | wc -c) + 1 ))
  echo "=== $name: serve on GPU $gpus port $port (tp=$ntp) ==="
  ssh kiti "tmux kill-session -t eval_$name 2>/dev/null; tmux new-session -d -s eval_$name \
    'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; \
     CUDA_VISIBLE_DEVICES=$gpus ~/micromamba/envs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
     --model $hf --served-model-name $name --tensor-parallel-size $ntp \
     --max-model-len 8192 --gpu-memory-utilization 0.92 --port $port --trust-remote-code \
     > ~/serve_eval_$name.log 2>&1'"
  up=0
  for i in $(seq 1 90); do
    if ssh kiti "curl -s -m 5 localhost:$port/v1/models" 2>/dev/null | grep -q "$name"; then up=1; break; fi
    sleep 10
  done
  if [ "$up" != 1 ]; then
    echo "SERVER_FAIL $name"; ssh kiti "tail -5 ~/serve_eval_$name.log 2>/dev/null; tmux kill-session -t eval_$name 2>/dev/null"
    continue
  fi

  tunnel "1$port" "$port"
  PYTHONUNBUFFERED=1 python3 eval/run_eval_v2.py \
    --base "http://localhost:1$port/v1" --model "$name" --key EMPTY \
    --workers 8 2>&1 | tee "eval/results_v2/${name}.log"

  ssh kiti "tmux kill-session -t eval_$name 2>/dev/null"
  if [ "$keep" != 1 ]; then
    echo "=== $name: evicting weights ==="
    ssh kiti "rm -rf $repo_dir"
  fi
done
echo "TEXT_SERIAL_V2_DONE"

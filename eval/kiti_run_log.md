# GEB-Bench kiti eval run log

Run started 2026-07-16 (Mac local time). GPUs 4,5,6,7 only (CUDA_VISIBLE_DEVICES=4,5,6,7). tensor-parallel 4, bf16.

## Environment
- 2026-07-16 ~16:40 — Verified kiti: driver 580.95.05 (CUDA 13.0), GPUs 4-7 idle (2MiB), /home 158G free.
- Launched tmux `vllm2setup`: micromamba env vllm2 (py3.12) + latest vllm. Log: ~/vllm2_pip.log
- Proxy via `source ~/export.sh` works (http://100.73.92.22:7890).
- No HF token on kiti → google/gemma-3-27b-it (gated) replaced by unsloth/gemma-3-27b-it (ungated bf16 mirror). Other repos verified: Qwen/Qwen2.5-VL-32B-Instruct, Qwen/Qwen3-32B, deepseek-ai/DeepSeek-R1-Distill-Qwen-32B.
- Deleted stale HF caches from previous (already-recorded) evals to free ~59G: Qwen2.5-7B-Instruct, Qwen2-VL-7B-Instruct, Qwen2.5-14B-Instruct.

- ~16:42 — micromamba binary not at ~/micromamba/bin (nix store path from .zshrc); created env with MAMBA_ROOT_PREFIX set. Python 3.12.13 OK.
- ~16:45 — `pip install vllm` (latest = 0.25.1) FAILED: kiti is Ubuntu 18.04 / glibc 2.27; torch >= 2.7 wheels are manylinux_2_28. Pinned vllm==0.8.5.post1 (torch 2.6.0), the newest glibc-2.27-compatible release; supports Qwen2.5-VL, Qwen3, Gemma-3.
- ~16:44 — Qwen2.5-VL-32B-Instruct download started in tmux `dl` (proxy, direct HF endpoint works fine).

- ~17:00 — vllm==0.8.5.post1 install failed: dep xformers==0.0.30 ships only manylinux_2_28 wheels (needs glibc>=2.28). Workaround: install vllm --no-deps + all other deps from wheel METADATA; rely on vLLM's bundled FlashAttention backend (Ampere default), xformers unused.
- ~17:05 — hf-mirror.com unusable from kiti (bytes trickle). Sticking with proxy (~6MB/s); restarted download with HF_HUB_ENABLE_HF_TRANSFER=1 for parallel chunks.

- 17:22 — vllm2 env ready: vLLM 0.8.5.post1, torch 2.6.0+cu124, python 3.12.13 (vllm --no-deps + 55 deps minus xformers; import verified).

## Models

#!/usr/bin/env bash
# Common env + args shared between the spec and non-spec runs.
# Mirrors the brutus vllm-xpu-chat service config minus the speculativeConfig
# block, so the two runs differ only in spec-decode being on/off.
set -euo pipefail

export VLLM_BIN="${VLLM_BIN:-/nix/store/n468660nf6h6p3b1vgc7fs4pbz44j6zm-python3.12-vllm-xpu-0.20.2.dev0+xpu.unstable/bin/vllm}"
export HF_HOME="${HF_HOME:-/var/cache/huggingface}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/tmp/vllm-spec-decode/cache}"
mkdir -p "$VLLM_CACHE_ROOT"
export VLLM_TARGET_DEVICE=xpu
export VLLM_XPU_ENABLE_XPU_GRAPH=1
export CCL_ATL_TRANSPORT=ofi
export CCL_LOG_LEVEL=warn
export CCL_PROCESS_LAUNCHER=none
export CCL_ZE_IPC_EXCHANGE=sockets

MODEL="palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4"
SERVED_NAME="qwen3.6-35b-a3b"
PORT=8000

COMMON_ARGS=(
  serve "$MODEL"
  --host 127.0.0.1
  --port "$PORT"
  --served-model-name "$SERVED_NAME"
  --dtype bfloat16
  --gpu-memory-utilization 0.83
  --quantization inc
  --kv-cache-dtype turboquant_k3v4_nc
  --max-model-len 65536
  --max-num-seqs 32
  --compilation-config '{"cudagraph_capture_sizes":[1,4]}'
  --language-model-only
)

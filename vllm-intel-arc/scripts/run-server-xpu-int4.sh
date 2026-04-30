#!/usr/bin/env bash
# Launch vLLM server with the in-progress XPU INT4 MoE path.
# Overlays our 3 modified Python files (xpu_moe.py, int_wna16.py, inc.py)
# from /home/jasonbk/Projects/vllm into the working baseline image
# vllm-xpu-tq:hybrid-noskip-4088d3dd0 via bind mounts.
#
# Modes:
#   plain  — INC INT4 MoE only (debug first boot)
#   tq     — INC INT4 MoE + TurboQuant k8v4 KV cache
#
# Usage: run-server-xpu-int4.sh [plain|tq] [-- <extra vllm args>]
set -euo pipefail

MODE="${1:-tq}"
shift || true

IMAGE="${VLLM_IMAGE:-vllm-xpu-tq:hybrid-noskip-4088d3dd0}"
MODEL="${MODEL:-palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4}"
PORT="${PORT:-8000}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-xpu-int4}"

VLLM_SRC="/home/jasonbk/Projects/vllm/vllm"
SITE="/opt/venv/lib/python3.12/site-packages/vllm"

HF_CACHE="$HOME/.cache/huggingface"
mkdir -p "$HF_CACHE"

# Stop existing container if running
podman rm -f "$CONTAINER_NAME" 2>/dev/null || true

# Build mode-specific vLLM args
# - plain:     INT4 MoE only, debug-friendly (max-num-seqs=1, eager) — FP16 KV
# - tq-k8v4:   + 8-bit K + 4-bit V        (~6 bits/elem, 2.6× vs FP16)
# - tq-4nc:    + 4-bit K + 4-bit V + NC   (4 bits/elem, 3.8×)
# - tq-k3v4nc: + 3-bit K + 4-bit V + NC   (~3.5×)
# - tq-3nc:    + 3-bit K + 3-bit V + NC   (3 bits/elem, 4.9×)
# - perf-*:    same KV mode but max-num-seqs=64 for the concurrency sweep
case "$MODE" in
    plain)
        MODE_ARGS=(--enforce-eager --max-num-seqs 1)
        ;;
    tq-k8v4)
        MODE_ARGS=(--kv-cache-dtype turboquant_k8v4 --enforce-eager --max-num-seqs 1)
        ;;
    tq-4nc)
        MODE_ARGS=(--kv-cache-dtype turboquant_4bit_nc --enforce-eager --max-num-seqs 1)
        ;;
    tq-k3v4nc)
        MODE_ARGS=(--kv-cache-dtype turboquant_k3v4_nc --enforce-eager --max-num-seqs 1)
        ;;
    tq-3nc)
        MODE_ARGS=(--kv-cache-dtype turboquant_3bit_nc --enforce-eager --max-num-seqs 1)
        ;;
    perf-k8v4)
        MODE_ARGS=(--kv-cache-dtype turboquant_k8v4 --enforce-eager --max-num-seqs 64)
        ;;
    perf-4nc)
        MODE_ARGS=(--kv-cache-dtype turboquant_4bit_nc --enforce-eager --max-num-seqs 64)
        ;;
    perf-3nc)
        MODE_ARGS=(--kv-cache-dtype turboquant_3bit_nc --enforce-eager --max-num-seqs 64)
        ;;
    # Legacy aliases
    tq)   MODE_ARGS=(--kv-cache-dtype turboquant_k8v4 --enforce-eager --max-num-seqs 1) ;;
    perf) MODE_ARGS=(--kv-cache-dtype turboquant_k8v4 --enforce-eager --max-num-seqs 64) ;;
    *)
        echo "Unknown mode: $MODE" >&2
        echo "Expected: plain|tq-k8v4|tq-4nc|tq-k3v4nc|tq-3nc|perf-k8v4|perf-4nc|perf-3nc" >&2
        exit 1
        ;;
esac

EXTRA_ARGS=("$@")

echo "Starting vLLM XPU INT4 MoE server"
echo "  Image:     $IMAGE"
echo "  Model:     $MODEL"
echo "  Mode:      $MODE"
echo "  Port:      $PORT"
echo "  Container: $CONTAINER_NAME"
echo ""

# --enforce-eager skips graph compilation (faster boot, easier debugging)
# --max-num-seqs 1 keeps memory footprint low for first-boot validation
# --limit-mm-per-prompt disables vision/audio (Qwen3.6-35B-A3B is text-only)
exec podman run --rm \
    --name "$CONTAINER_NAME" \
    --device /dev/dri \
    --group-add keep-groups \
    --ipc=host \
    -p "$PORT:8000" \
    -v "$HF_CACHE:/root/.cache/huggingface:Z" \
    -v "$VLLM_SRC/model_executor/layers/fused_moe/experts/xpu_moe.py:$SITE/model_executor/layers/fused_moe/experts/xpu_moe.py:ro,Z" \
    -v "$VLLM_SRC/model_executor/layers/fused_moe/oracle/int_wna16.py:$SITE/model_executor/layers/fused_moe/oracle/int_wna16.py:ro,Z" \
    -v "$VLLM_SRC/model_executor/layers/quantization/inc.py:$SITE/model_executor/layers/quantization/inc.py:ro,Z" \
    --entrypoint /bin/bash \
    "$IMAGE" \
    -c "source /root/.bashrc && exec vllm serve '$MODEL' \
        --quantization inc \
        --dtype bfloat16 \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.85 \
        --max-logprobs ${MAX_LOGPROBS:-2000} \
        --limit-mm-per-prompt '{\"image\":0,\"video\":0}' \
        --host 0.0.0.0 --port 8000 \
        ${MODE_ARGS[*]} ${EXTRA_ARGS[*]}"

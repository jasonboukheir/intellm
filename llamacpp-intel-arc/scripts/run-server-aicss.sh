#!/usr/bin/env bash
# Run a llama-server built from the aicss-genai/llama.cpp fork (#22066 patches
# applied). Re-uses the intel/vllm container as the runtime image because it
# already has oneAPI 2025.3 + level-zero + intel-compute-runtime, which are
# what the patched binary needs at run time.
#
# Build with: scripts/build-aicss.sh   (binary lands at build-aicss/llama.cpp/build/bin/)
#
# Usage:
#   scripts/run-server-aicss.sh [path/to/config.yaml] [-- extra llama-server args]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNTIME_IMAGE="${LLAMA_AICSS_RUNTIME:-docker.io/intel/vllm:0.17.0-xpu}"
BIN_DIR="$ROOT/build-aicss/llama-pr-only/build/bin"
CONFIG="${1:-$ROOT/configs/models/qwen3.6-35b-a3b-q4km.yaml}"
PORT="${PORT:-8081}"
shift || true

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [ ! -f "$CONFIG" ]; then
    echo "config not found: $CONFIG" >&2; exit 1
fi
if [ ! -x "$BIN_DIR/llama-server" ]; then
    echo "patched llama-server not built: $BIN_DIR/llama-server" >&2
    echo "run scripts/build-aicss.sh first" >&2
    exit 1
fi

get() {
    local key="$1" default="$2"
    if command -v yq >/dev/null 2>&1; then
        local v; v=$(yq -r ".${key} // \"\"" "$CONFIG")
        [ -n "$v" ] && [ "$v" != "null" ] && echo "$v" || echo "$default"
    else
        local v
        v=$(grep -E "^${key}:" "$CONFIG" | head -1 | sed -E "s/^${key}:[[:space:]]*//" | tr -d '"' | sed -E 's/[[:space:]]+#.*$//')
        [ -n "$v" ] && echo "$v" || echo "$default"
    fi
}

GGUF=$(get gguf_file "")
ALIAS=$(get served_model_alias "$GGUF")
CTX=$(get context_size 4096)
NGL=$(get n_gpu_layers -1)
NB=$(get n_batch 2048)
NUB=$(get n_ubatch 512)
PAR=$(get parallel 1)
FA=$(get flash_attn true)
THREADS=$(get threads 8)

CACHE="$HOME/.cache/llamacpp/models"
[ -f "$CACHE/$GGUF" ] || { echo "missing $CACHE/$GGUF — run pull-model.sh first" >&2; exit 1; }

FLASH_ARGS=(--flash-attn auto)
case "$FA" in
    true|True|on)  FLASH_ARGS=(--flash-attn on)  ;;
    false|False|off) FLASH_ARGS=(--flash-attn off) ;;
esac

CONTAINER_NAME="${CONTAINER_NAME:-llamacpp-aicss}"

echo "=== llama-server (aicss-genai patched) ==="
echo "  Runtime image: $RUNTIME_IMAGE"
echo "  Binary:        $BIN_DIR/llama-server"
echo "  Model:         $CACHE/$GGUF"
echo "  Port:          $PORT"
echo

# Note: AOT-compiled with bmg-g31; GGML_SYCL_DISABLE_OPT shouldn't be needed
# now (the AOT fix in PR #22147 supersedes the JIT corruption workaround).
# Set LLAMA_AICSS_DISABLE_OPT=1 to keep the workaround on if you see issues.
EXTRA_ENV=()
if [ "${LLAMA_AICSS_DISABLE_OPT:-0}" = "1" ]; then
    EXTRA_ENV=(-e GGML_SYCL_DISABLE_OPT=1)
fi

exec podman run --rm \
    --name "$CONTAINER_NAME" \
    --device /dev/dri \
    --group-add keep-groups \
    --shm-size=4g \
    -p "$PORT:8080" \
    -v "$BIN_DIR:/llama:z,ro" \
    -v "$CACHE:/models:z,ro" \
    -e SETVARS_COMPLETED=0 \
    "${EXTRA_ENV[@]}" \
    --entrypoint /bin/bash \
    "$RUNTIME_IMAGE" \
    -lc ". /opt/intel/oneapi/setvars.sh --force >/dev/null && \
        export LD_LIBRARY_PATH=/llama:\${LD_LIBRARY_PATH:-} && \
        exec /llama/llama-server \
        --model /models/$GGUF \
        --alias '$ALIAS' \
        --host 0.0.0.0 --port 8080 \
        --ctx-size $CTX \
        --n-gpu-layers $NGL \
        --batch-size $NB --ubatch-size $NUB \
        --parallel $PAR --threads $THREADS \
        ${FLASH_ARGS[*]} \
        --metrics \
        ${EXTRA_ARGS[*]}"

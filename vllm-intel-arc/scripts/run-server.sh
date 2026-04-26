#!/usr/bin/env bash
set -euo pipefail

IMAGE="${VLLM_IMAGE:-intel/vllm@sha256:e961d08135a6a8ef6decd857c6deab7a70eb00e19de21de54cbc0ce05d9a9f43}"
CONFIG_DIR="$(cd "$(dirname "$0")/../configs/models" && pwd)"
PORT="${PORT:-8000}"

usage() {
    echo "Usage: $0 [--config <config.yaml>] [--port <port>] [-- <extra vllm args>]"
    echo ""
    echo "Options:"
    echo "  --config   Path to model config YAML (default: configs/models/llama-3.1-8b.yaml)"
    echo "  --port     API server port (default: 8000)"
    echo ""
    echo "Available configs:"
    ls "$CONFIG_DIR"/*.yaml 2>/dev/null | while read f; do
        echo "  $(basename "$f")"
    done
    exit 1
}

CONFIG="$CONFIG_DIR/llama-3.1-8b.yaml"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --help|-h) usage ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ ! -f "$CONFIG" ]]; then
    echo "Config not found: $CONFIG"
    usage
fi

echo "Starting vLLM server"
echo "  Image:  $IMAGE"
echo "  Config: $CONFIG"
echo "  Port:   $PORT"
echo ""

# podman bind-mounts fail if the source directory does not yet exist
HF_CACHE="$HOME/.cache/huggingface"
mkdir -p "$HF_CACHE"

# Parse YAML config into vLLM CLI args
MODEL=$(yq -r '.model' "$CONFIG")
DTYPE=$(yq -r '.dtype // "auto"' "$CONFIG")
TP=$(yq -r '.tensor_parallel_size // 1' "$CONFIG")
MAX_LEN=$(yq -r '.max_model_len // 4096' "$CONFIG")
GPU_UTIL=$(yq -r '.gpu_memory_utilization // 0.90' "$CONFIG")
REVISION=$(yq -r '.revision // ""' "$CONFIG")

REVISION_ARGS=()
if [[ -n "$REVISION" ]]; then
    REVISION_ARGS=(--revision "$REVISION")
fi

# Run container with GPU passthrough.
# - keep-groups maps the host's render-group membership into the container,
#   which is what /dev/dri/renderD* needs (literal "render" GID often mismatches).
# - The base image has no ENTRYPOINT (CMD=/bin/bash), so we invoke `vllm serve`
#   explicitly. --device xpu selects the XPU backend.
CONTAINER_NAME="${CONTAINER_NAME:-vllm-server}"

exec podman run --rm \
    --name "$CONTAINER_NAME" \
    --device /dev/dri \
    --group-add keep-groups \
    --shm-size=16g \
    -p "$PORT:8000" \
    -v "$HF_CACHE:/root/.cache/huggingface:z" \
    "$IMAGE" \
    vllm serve "$MODEL" \
    --dtype "$DTYPE" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --host 0.0.0.0 \
    --port 8000 \
    "${REVISION_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"

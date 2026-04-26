#!/usr/bin/env bash
# Run llama-server (SYCL build) with the given model config, exposing an
# OpenAI-compatible API on $PORT.
#
# Usage:
#   scripts/run-server.sh [path/to/config.yaml] [-- extra llama-server args]
#
# The container is intentionally rootless + bind-mounts the model from
# ~/.cache/llamacpp/models. We pass --device /dev/dri and --group-add
# keep-groups so the render group on the host maps in correctly (same
# pattern vllm-intel-arc uses).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${LLAMA_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-intel}"
PORT="${PORT:-8080}"
CONFIG="${1:-$ROOT/configs/models/qwen3.6-35b-a3b-q4km.yaml}"
shift || true

# Anything after `--` is forwarded raw to llama-server.
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [ ! -f "$CONFIG" ]; then
    echo "config not found: $CONFIG" >&2
    exit 1
fi

# yq is provided by the project nix devshell; fall back to grep when invoked
# from the bare host shell.
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
if [ ! -f "$CACHE/$GGUF" ]; then
    echo "GGUF not found in cache: $CACHE/$GGUF" >&2
    echo "Run: nix run .#pull-model -- $CONFIG" >&2
    exit 1
fi

CONTAINER_NAME="${CONTAINER_NAME:-llamacpp-server}"

echo "Starting llama-server"
echo "  Image:   $IMAGE"
echo "  Model:   $CACHE/$GGUF"
echo "  Alias:   $ALIAS"
echo "  Port:    $PORT"
echo "  Context: $CTX"
echo "  Parallel: $PAR  (max concurrent sequences)"
echo

FLASH_ARGS=(--flash-attn auto)
case "$FA" in
    true|True|on)  FLASH_ARGS=(--flash-attn on)  ;;
    false|False|off) FLASH_ARGS=(--flash-attn off) ;;
esac

# GGML_SYCL_DISABLE_OPT=1 works around the Battlemage F16-opt corruption bug.
# See: https://github.com/ggml-org/llama.cpp/issues/21893
exec podman run --rm \
    --name "$CONTAINER_NAME" \
    --device /dev/dri \
    --group-add keep-groups \
    --shm-size=4g \
    -p "$PORT:8080" \
    -v "$CACHE:/models:z,ro" \
    -e GGML_SYCL_DISABLE_OPT=1 \
    "$IMAGE" \
    --model "/models/$GGUF" \
    --alias "$ALIAS" \
    --host 0.0.0.0 --port 8080 \
    --ctx-size "$CTX" \
    --n-gpu-layers "$NGL" \
    --batch-size "$NB" \
    --ubatch-size "$NUB" \
    --parallel "$PAR" \
    --threads "$THREADS" \
    "${FLASH_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"

#!/usr/bin/env bash
# Profile decode for llama.cpp/SYCL (parallels vllm-intel-arc/profile-decode.sh).
#
# llama-server's /metrics endpoint exposes Prometheus-style counters very
# similar to vLLM's, so we can do the same before/after delta to compute
# prefill vs decode time, tokens/s, and BW utilization. We don't have
# /v1/start_profile here either — engine-level metrics are sufficient to
# answer "is this still bandwidth-bound after Q4?"
#
# Usage:
#   scripts/profile-decode.sh [path/to/config.yaml]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${LLAMA_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-intel}"
CONFIG="${1:-$ROOT/configs/models/qwen3.6-35b-a3b-q4km.yaml}"
PORT="${PORT:-8080}"
BASE_URL="http://127.0.0.1:$PORT"
RESULTS_DIR="$ROOT/results/profile-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RESULTS_DIR"

CONTAINER_NAME="llamacpp-profile-$$"
CACHE="$HOME/.cache/llamacpp/models"

if [ ! -f "$CONFIG" ]; then
    echo "config not found: $CONFIG" >&2
    exit 1
fi
GGUF=$(yq -r '.gguf_file' "$CONFIG")
ALIAS=$(yq -r '.served_model_alias // .gguf_file' "$CONFIG")
CTX=$(yq -r '.context_size // 4096' "$CONFIG")
PAR=$(yq -r '.parallel // 1' "$CONFIG")
NGL=$(yq -r '.n_gpu_layers // -1' "$CONFIG")
NB=$(yq -r '.n_batch // 2048' "$CONFIG")
NUB=$(yq -r '.n_ubatch // 512' "$CONFIG")
THREADS=$(yq -r '.threads // 8' "$CONFIG")
FA=$(yq -r '.flash_attn // true' "$CONFIG")

if [ ! -f "$CACHE/$GGUF" ]; then
    echo "GGUF not found: $CACHE/$GGUF" >&2
    echo "Run: nix run .#pull-model -- $CONFIG" >&2
    exit 1
fi

cleanup() {
    echo "[cleanup] stopping $CONTAINER_NAME"
    [ -n "${LOGS_PID:-}" ] && kill "$LOGS_PID" 2>/dev/null || true
    podman stop -t 5 "$CONTAINER_NAME" 2>/dev/null || true
    podman rm -f "$CONTAINER_NAME" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# wait_pids: wait only on these PIDs (NOT on backgrounded podman logs -f).
wait_pids() { for p in "$@"; do wait "$p" || true; done; }

FLASH_ARGS=()
[ "$FA" = "true" ] && FLASH_ARGS=(--flash-attn)

echo "=== llama.cpp/SYCL workload profile ==="
echo "Model:       $GGUF (alias: $ALIAS)"
echo "Container:   $CONTAINER_NAME"
echo "Results dir: $RESULTS_DIR"
echo

echo "[1/6] starting server..."
podman run -d --rm \
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
    --metrics \
    "${FLASH_ARGS[@]}" > /dev/null

podman logs -f "$CONTAINER_NAME" > "$RESULTS_DIR/server.log" 2>&1 &
LOGS_PID=$!

echo "[2/6] waiting for /v1/models (up to 8 minutes)..."
WAIT_S=480
for i in $(seq 1 "$WAIT_S"); do
    if curl -sf -o /dev/null --max-time 2 "$BASE_URL/v1/models"; then
        echo "      ready after ${i}s."; break
    fi
    if ! podman ps --filter "name=$CONTAINER_NAME" --format '{{.Names}}' | grep -q "$CONTAINER_NAME"; then
        echo "      container exited unexpectedly. Tail of server.log:" >&2
        tail -80 "$RESULTS_DIR/server.log" >&2
        exit 1
    fi
    if [ "$i" -eq "$WAIT_S" ]; then
        echo "      server not ready after ${WAIT_S}s. Tail of server.log:" >&2
        tail -80 "$RESULTS_DIR/server.log" >&2
        exit 1
    fi
    sleep 1
done

curl -sf "$BASE_URL/metrics" > "$RESULTS_DIR/metrics-before.txt" || true

PROMPT='In a recent interview about distributed systems, Dr. Kim explained that consistency models matter most when you '
PROMPT="$PROMPT$(printf 'consider %s. ' {1..120})"

send_request() {
    local max_tokens="$1"
    curl -sf -X POST "$BASE_URL/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"$ALIAS\", \"prompt\": \"$PROMPT\", \"max_tokens\": $max_tokens, \"temperature\": 0.0}" \
        > /dev/null
}

echo "[3/6] warmup..."
send_request 32

echo "[4/6] running workloads..."
echo "      single-stream, 256 tokens..."
send_request 256
echo "      4 concurrent, 128 tokens..."
pids=()
for _ in 1 2 3 4; do send_request 128 & pids+=($!); done
wait_pids "${pids[@]}"
echo "      8 concurrent, 64 tokens..."
pids=()
for _ in 1 2 3 4 5 6 7 8; do send_request 64 & pids+=($!); done
wait_pids "${pids[@]}"

echo "[5/6] snapshotting /metrics (after)..."
curl -sf "$BASE_URL/metrics" > "$RESULTS_DIR/metrics-after.txt" || true

echo "[6/6] extracting timing/throughput from server log..."
grep -E "prompt eval|eval time|tokens per second|slot.*released|slot.*processing" \
    "$RESULTS_DIR/server.log" > "$RESULTS_DIR/engine-stats.txt" 2>/dev/null || true

echo
echo "  Server log:    $RESULTS_DIR/server.log"
echo "  Engine stats:  $RESULTS_DIR/engine-stats.txt   ($(wc -l < "$RESULTS_DIR/engine-stats.txt" 2>/dev/null || echo 0) lines)"
echo "  Metrics:       $RESULTS_DIR/metrics-{before,after}.txt"

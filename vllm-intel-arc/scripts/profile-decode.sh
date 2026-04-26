#!/usr/bin/env bash
# Profile vLLM by parsing the engine's per-step stats and Prometheus metrics.
#
# We tried the PyTorch profiler path (VLLM_TORCH_PROFILER_DIR + /start_profile)
# but the intel/vllm:0.17.0-xpu build doesn't recognize the env var nor the
# endpoint. Engine stats are sufficient to answer the macro question
# ("decode memory-bound?"): vLLM logs per-iteration prefill_throughput and
# generation_throughput, and /metrics has KV-cache utilization, request
# queue depth, and token-level timings.
#
# Workflow:
#   1. Start vLLM server, stream container logs to file.
#   2. Wait for /v1/models.
#   3. Snapshot /metrics (before).
#   4. Run workloads at several concurrencies/prompt lengths to populate stats.
#   5. Snapshot /metrics (after).
#   6. Extract engine-stats lines from server.log into stats.txt.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${VLLM_IMAGE:-intel/vllm@sha256:e961d08135a6a8ef6decd857c6deab7a70eb00e19de21de54cbc0ce05d9a9f43}"
PORT="${PORT:-8000}"
BASE_URL="http://127.0.0.1:$PORT"
RESULTS_DIR="$ROOT/results/profile-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RESULTS_DIR"

CONTAINER_NAME="vllm-profile-$$"
HF_CACHE="$HOME/.cache/huggingface"
mkdir -p "$HF_CACHE"

# Inlined for the Llama-3.1-8B baseline. Override with env vars if needed.
# (We previously read these from configs/models/*.yaml via yq, but the host
# shell doesn't have yq outside of the nix devshell; inlining avoids the
# dependency for this profiling script.)
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
DTYPE="${DTYPE:-bfloat16}"
TP="${TP:-1}"
MAX_LEN="${MAX_LEN:-8192}"
GPU_UTIL="${GPU_UTIL:-0.90}"

cleanup() {
    echo "[cleanup] stopping $CONTAINER_NAME"
    [ -n "${LOGS_PID:-}" ] && kill "$LOGS_PID" 2>/dev/null || true
    podman stop -t 5 "$CONTAINER_NAME" 2>/dev/null || true
    podman rm -f "$CONTAINER_NAME" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "=== vLLM workload profile ==="
echo "Model:       $MODEL ($DTYPE)"
echo "Container:   $CONTAINER_NAME"
echo "Results dir: $RESULTS_DIR"
echo

echo "[1/6] starting server..."
podman run -d --rm \
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
        --host 0.0.0.0 --port 8000 > /dev/null

# Stream container logs to file in the background so we can debug failures.
podman logs -f "$CONTAINER_NAME" > "$RESULTS_DIR/server.log" 2>&1 &
LOGS_PID=$!

echo "[2/6] waiting for /v1/models (up to 8 minutes; first start can take ~3 min)..."
WAIT_S=480
for i in $(seq 1 "$WAIT_S"); do
    if curl -sf -o /dev/null --max-time 2 "$BASE_URL/v1/models"; then
        echo "      ready after ${i}s."
        break
    fi
    # If the container died, bail early.
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

# Snapshot metrics before workload
curl -sf "$BASE_URL/metrics" > "$RESULTS_DIR/metrics-before.txt" || true

# Build a real-ish prompt; ~1024 tokens of context to match a typical decode-heavy workload
PROMPT='In a recent interview about distributed systems, Dr. Kim explained that consistency models matter most when you '
PROMPT="$PROMPT$(printf 'consider %s. ' {1..120})"

send_request() {
    local max_tokens="$1"
    curl -sf -X POST "$BASE_URL/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"$MODEL\", \"prompt\": \"$PROMPT\", \"max_tokens\": $max_tokens, \"temperature\": 0.0}" \
        > /dev/null
}

# wait_pids: wait only on the curl PIDs (NOT on `podman logs -f`, which would
# block forever).
wait_pids() {
    for p in "$@"; do wait "$p" || true; done
}

echo "[3/6] warmup..."
send_request 32

echo "[4/6] running workloads (single-stream + 4-concurrency, decode-heavy)..."
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

echo "[6/6] extracting engine stats..."
grep -E "Engine .*: Avg prompt throughput|Avg prompt|Avg generation|Running:|Pending:|GPU KV cache" \
    "$RESULTS_DIR/server.log" > "$RESULTS_DIR/engine-stats.txt" 2>/dev/null || true
echo
echo "  Server log:    $RESULTS_DIR/server.log"
echo "  Engine stats:  $RESULTS_DIR/engine-stats.txt   ($(wc -l < "$RESULTS_DIR/engine-stats.txt" 2>/dev/null || echo 0) lines)"
echo "  Metrics:       $RESULTS_DIR/metrics-{before,after}.txt"
echo
echo "Run: scripts/summarize-metrics.sh $RESULTS_DIR/"

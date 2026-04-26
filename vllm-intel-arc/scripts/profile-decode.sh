#!/usr/bin/env bash
# Profile vLLM decode with PyTorch profiler (medium-depth profile).
#
# Workflow:
#   1. Start the vLLM server in the background with profiler enabled
#      (VLLM_TORCH_PROFILER_DIR=/tmp/vllm-traces inside the container).
#   2. Wait for /v1/models.
#   3. Send a warmup request so the engine is past compilation.
#   4. POST /start_profile to begin tracing.
#   5. Send N short generation requests so we capture decode steps.
#   6. POST /stop_profile to dump the trace.
#   7. Stop the server. The trace JSON ends up in $RESULTS_DIR.
#
# The trace can be opened in chrome://tracing or perfetto.dev.
#
# We also poll /metrics during the workload so we get vLLM's own counters
# (token/s, kv-cache util) alongside the kernel timeline.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${VLLM_IMAGE:-intel/vllm@sha256:e961d08135a6a8ef6decd857c6deab7a70eb00e19de21de54cbc0ce05d9a9f43}"
PORT="${PORT:-8000}"
BASE_URL="http://127.0.0.1:$PORT"
RESULTS_DIR="$ROOT/results/profile-$(date +%Y%m%d-%H%M%S)"
TRACE_DIR_HOST="$RESULTS_DIR/traces"
mkdir -p "$TRACE_DIR_HOST"

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
    podman stop -t 5 "$CONTAINER_NAME" 2>/dev/null || true
    podman rm -f "$CONTAINER_NAME" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "=== vLLM decode profile ==="
echo "Model:       $MODEL ($DTYPE)"
echo "Container:   $CONTAINER_NAME"
echo "Trace dir:   $TRACE_DIR_HOST"
echo

echo "[1/7] starting server with VLLM_TORCH_PROFILER_DIR enabled..."
podman run -d --rm \
    --name "$CONTAINER_NAME" \
    --device /dev/dri \
    --group-add keep-groups \
    --shm-size=16g \
    -p "$PORT:8000" \
    -v "$HF_CACHE:/root/.cache/huggingface:z" \
    -v "$TRACE_DIR_HOST:/traces:z" \
    -e VLLM_TORCH_PROFILER_DIR=/traces \
    "$IMAGE" \
    vllm serve "$MODEL" \
        --dtype "$DTYPE" \
        --tensor-parallel-size "$TP" \
        --max-model-len "$MAX_LEN" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --host 0.0.0.0 --port 8000 \
        > "$RESULTS_DIR/server.log" 2>&1

echo "[2/7] waiting for /v1/models (up to 5 minutes)..."
for i in $(seq 1 300); do
    if curl -sf -o /dev/null --max-time 2 "$BASE_URL/v1/models"; then
        echo "      ready after ${i}s."
        break
    fi
    if [ "$i" -eq 300 ]; then
        echo "      server not ready after 300s. Server log:" >&2
        tail -50 "$RESULTS_DIR/server.log" >&2
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

echo "[3/7] warmup (compile path / cuda graphs / etc.)..."
send_request 32

echo "[4/7] starting torch profiler..."
curl -sf -X POST "$BASE_URL/start_profile"
sleep 1

echo "[5/7] running profiled workload (4 concurrent requests, 64 tokens each)..."
for _ in 1 2 3 4; do
    send_request 64 &
done
wait

echo "[6/7] stopping torch profiler (this can take ~30s as it dumps the trace)..."
curl -sf -X POST "$BASE_URL/stop_profile"

# Wait for trace files to actually appear (the dump is async)
echo "      waiting for trace JSON to appear..."
for i in $(seq 1 60); do
    if ls "$TRACE_DIR_HOST"/*.pt.trace.json* "$TRACE_DIR_HOST"/*.json 2>/dev/null | head -1 >/dev/null; then
        break
    fi
    sleep 2
done

# Snapshot metrics after workload
curl -sf "$BASE_URL/metrics" > "$RESULTS_DIR/metrics-after.txt" || true

echo "[7/7] summary"
echo "  Server log:        $RESULTS_DIR/server.log"
echo "  Trace files:       $TRACE_DIR_HOST/"
ls -lh "$TRACE_DIR_HOST/" 2>/dev/null || true
echo "  Metrics snapshot:  $RESULTS_DIR/metrics-{before,after}.txt"
echo
echo "Open trace in https://ui.perfetto.dev or chrome://tracing"
echo "Then run: scripts/summarize-profile.py $TRACE_DIR_HOST/<trace.json>"

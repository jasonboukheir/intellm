#!/usr/bin/env bash
set -euo pipefail

# Full baseline collection: throughput + quality on Qwen2.5-7B-Instruct
#
# Prerequisites:
#   1. GPU detected (run nix-intel-xpu/tests/smoke-test.sh)
#   2. Container pulled (./scripts/pull-container.sh)
#   3. HuggingFace token set (for gated models)
#
# This script:
#   1. Starts vLLM with Qwen2.5-7B at BF16 (quality eval config)
#   2. Runs throughput benchmarks
#   3. Captures baseline logprobs on WikiText-2
#   4. Stops the server
#
# Results land in results/

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PORT="${PORT:-8000}"

echo "=== Full Baseline Collection ==="
echo "Model: Qwen/Qwen2.5-7B-Instruct (BF16)"
echo "Port:  $PORT"
echo ""

# Start server in background
echo "--- Starting vLLM server ---"
"$SCRIPT_DIR/run-server.sh" \
    --config "$PROJECT_DIR/configs/models/qwen2.5-7b-quality-eval.yaml" \
    --port "$PORT" &
SERVER_PID=$!

cleanup() {
    echo ""
    echo "Stopping server (PID $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for server readiness
echo "Waiting for server..."
for i in $(seq 1 120); do
    if curl -s "http://localhost:$PORT/health" >/dev/null 2>&1; then
        echo "Server ready after ${i}s."
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Server process died. Check container logs."
        exit 1
    fi
    if [ "$i" -eq 120 ]; then
        echo "Server not ready after 120s, aborting."
        exit 1
    fi
    sleep 1
done

MODEL=$(curl -s "http://localhost:$PORT/v1/models" | jq -r '.data[0].id')
echo "Model loaded: $MODEL"
echo ""

# Step 1: Throughput benchmarks
echo "--- Throughput Benchmarks ---"
mkdir -p "$PROJECT_DIR/results"
python3 "$PROJECT_DIR/benchmarks/run_benchmarks.py" \
    --base-url "http://localhost:$PORT" \
    --model "$MODEL" \
    --output "$PROJECT_DIR/results/throughput_baseline.json"
echo ""

# Step 2: Quality baseline (WikiText-2 logprobs)
echo "--- Quality Baseline (WikiText-2) ---"
python3 "$PROJECT_DIR/benchmarks/quality/capture_logprobs.py" \
    --base-url "http://localhost:$PORT" \
    --model "$MODEL" \
    --tag baseline \
    --dataset wikitext2 \
    --max-sequences 100 \
    --max-tokens 512 \
    --output "$PROJECT_DIR/results/quality"
echo ""

echo "=== Baseline Collection Complete ==="
echo "Results:"
echo "  Throughput: results/throughput_baseline.json"
echo "  Quality:    results/quality/logprobs_baseline.json"
echo ""
echo "Next steps:"
echo "  1. Enable RotorQuant KV cache compression"
echo "  2. Run: ./scripts/run-quality-eval.sh rotorquant-3bit"
echo "  3. Compare: python benchmarks/quality/compute_kl.py \\"
echo "       results/quality/logprobs_baseline.json \\"
echo "       results/quality/logprobs_rotorquant-3bit.json"

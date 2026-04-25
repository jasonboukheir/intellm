#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8000}"
BASE_URL="http://localhost:$PORT"
RESULTS_DIR="$(cd "$(dirname "$0")/.." && pwd)/results"
mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)

echo "=== vLLM Intel Arc Benchmark ==="
echo "Server: $BASE_URL"
echo "Results: $RESULTS_DIR"
echo ""

# Wait for server (vLLM exposes /v1/models, not /health)
echo "Waiting for server..."
for i in $(seq 1 60); do
    if curl -sf -o /dev/null --max-time 2 "$BASE_URL/v1/models"; then
        echo "Server ready."
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "Server not ready after 60s, aborting."
        exit 1
    fi
    sleep 1
done

# Get model info
MODEL=$(curl -s "$BASE_URL/v1/models" | jq -r '.data[0].id')
echo "Model: $MODEL"
echo ""

# Run benchmark
python3 "$(dirname "$0")/../benchmarks/run_benchmarks.py" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --output "$RESULTS_DIR/benchmark-${TIMESTAMP}.json" \
    "$@"

#!/usr/bin/env bash
# Throughput benchmark against a running llama-server. Mirrors the vllm-intel-arc
# benchmark (same prompt-len × output-len × concurrency grid) so results are
# directly comparable.
set -euo pipefail

PORT="${PORT:-8080}"
BASE_URL="http://127.0.0.1:$PORT"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="$ROOT/results"
mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
TAG="${TAG:-baseline}"

echo "=== llama.cpp/SYCL benchmark ==="
echo "Server:  $BASE_URL"
echo "Tag:     $TAG"
echo "Results: $RESULTS_DIR"

echo "Waiting for server..."
for i in $(seq 1 60); do
    # llama-server exposes /v1/models (OpenAI compat) once ready.
    if curl -sf -o /dev/null --max-time 2 "$BASE_URL/v1/models"; then
        echo "Server ready."; break
    fi
    if [ "$i" -eq 60 ]; then
        echo "Server not ready after 60s, aborting." >&2
        exit 1
    fi
    sleep 1
done

MODEL=$(curl -s "$BASE_URL/v1/models" | jq -r '.data[0].id')
echo "Model: $MODEL"
echo

# Reuse the vllm benchmark harness — it speaks /v1/completions, which
# llama-server implements compatibly.
exec python3 "$ROOT/../vllm-intel-arc/benchmarks/run_benchmarks.py" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --output "$RESULTS_DIR/benchmark-${TAG}-${TIMESTAMP}.json" \
    "$@"

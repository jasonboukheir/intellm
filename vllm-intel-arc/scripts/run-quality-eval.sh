#!/usr/bin/env bash
set -euo pipefail

# Full quality evaluation pipeline:
#   1. Capture baseline logprobs (BF16 KV cache)
#   2. [swap in compressed KV cache, restart server]
#   3. Capture comparison logprobs
#   4. Compute KL divergence + perplexity delta
#
# This script runs one capture. Run it twice with different --tag values,
# then use compute_kl.py to compare.

PORT="${PORT:-8000}"
BASE_URL="http://localhost:$PORT"
RESULTS_DIR="$(cd "$(dirname "$0")/.." && pwd)/results/quality"
BENCH_DIR="$(cd "$(dirname "$0")/../benchmarks/quality" && pwd)"
mkdir -p "$RESULTS_DIR"

TAG="${1:?Usage: $0 <tag> [dataset] [max-sequences]}"
DATASET="${2:-wikitext2}"
MAX_SEQ="${3:-100}"

echo "=== Quality Evaluation ==="
echo "Tag:      $TAG"
echo "Dataset:  $DATASET"
echo "Max seq:  $MAX_SEQ"
echo "Server:   $BASE_URL"
echo ""

# Wait for server
echo "Waiting for server..."
for i in $(seq 1 60); do
    if curl -s "$BASE_URL/health" >/dev/null 2>&1; then
        echo "Server ready."
        break
    fi
    if [ "$i" -eq 60 ]; then echo "Server not ready, aborting."; exit 1; fi
    sleep 1
done

MODEL=$(curl -s "$BASE_URL/v1/models" | jq -r '.data[0].id')
echo "Model: $MODEL"
echo ""

# Capture logprobs
python3 "$BENCH_DIR/capture_logprobs.py" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --tag "$TAG" \
    --dataset "$DATASET" \
    --max-sequences "$MAX_SEQ" \
    --output "$RESULTS_DIR"

echo ""
echo "Capture complete: $RESULTS_DIR/logprobs_${TAG}.json"

# If both baseline and this tag exist, auto-compare
BASELINE="$RESULTS_DIR/logprobs_baseline.json"
CAPTURE="$RESULTS_DIR/logprobs_${TAG}.json"
if [ "$TAG" != "baseline" ] && [ -f "$BASELINE" ] && [ -f "$CAPTURE" ]; then
    echo ""
    echo "=== Auto-comparing against baseline ==="
    python3 "$BENCH_DIR/compute_kl.py" \
        "$BASELINE" "$CAPTURE" \
        --output "$RESULTS_DIR/kl_baseline_vs_${TAG}.json"
fi

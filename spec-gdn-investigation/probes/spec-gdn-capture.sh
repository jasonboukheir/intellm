#!/usr/bin/env bash
# Capture runner for the spec-aware SYCL gdn_attention loop.
# Runs INSIDE the vllm-dev container (via `vllm-run bash <this>`),
# launches vllm serve with FLA forced + dump dir set, drives a 12-prompt
# suite, verifies tuple count and flavor mix, kills the server.
#
# Outputs:
#   /tmp/spec_gdn_serve.log       — server stdout/stderr
#   /tmp/spec_gdn_captures/*.pt   — captured tuples (consumed by replay test)
#   /tmp/spec_gdn_capture.summary — final pass/fail summary

set -uo pipefail

CAPTURES_DIR=/tmp/spec_gdn_captures
SERVE_LOG=/tmp/spec_gdn_serve.log
SUMMARY=/tmp/spec_gdn_capture.summary
MODEL=palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4
PORT=8040

mkdir -p "$CAPTURES_DIR"
rm -f "$CAPTURES_DIR"/tuple_*.pt
: > "$SERVE_LOG"
: > "$SUMMARY"

echo "[capture] $(date -Iseconds) launching vllm serve" | tee -a "$SUMMARY"

VLLM_XPU_FORCE_FLA_GDN=1 \
VLLM_XPU_DUMP_SPEC_GDN="$CAPTURES_DIR" \
VLLM_XPU_DUMP_SPEC_GDN_MAX=200 \
vllm serve "$MODEL" \
  --tensor-parallel-size 1 \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":3}' \
  --trust-remote-code \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --port "$PORT" > "$SERVE_LOG" 2>&1 &

SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  for _ in $(seq 1 30); do
    kill -0 "$SERVER_PID" 2>/dev/null || break
    sleep 1
  done
  kill -9 "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "[capture] server pid=$SERVER_PID; waiting for /health (up to 1200s)" | tee -a "$SUMMARY"
ready=0
for i in $(seq 1 1200); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    ready=1
    echo "[capture] server READY at +${i}s" | tee -a "$SUMMARY"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[capture] server DIED before /health responded" | tee -a "$SUMMARY"
    tail -80 "$SERVE_LOG" | tee -a "$SUMMARY"
    exit 2
  fi
  sleep 1
done

if [ "$ready" -ne 1 ]; then
  echo "[capture] FAIL: server didn't reach /health in 1200s" | tee -a "$SUMMARY"
  tail -80 "$SERVE_LOG" | tee -a "$SUMMARY"
  exit 3
fi

PROMPTS=(
  "What's the capital of France?"
  "Explain quantum entanglement in two short sentences."
  "Write a haiku about a debugger finding a race condition."
  "List five prime numbers between 100 and 200, comma separated."
  "Translate the phrase 'good morning' into Japanese, with romaji."
  "What is the time complexity of merge sort and why?"
  "Summarize the plot of Hamlet in two sentences."
  "Name three Linux kernel scheduler classes and a one-line tradeoff each."
  "Why is the sky blue? Answer in two sentences."
  "Write a short Python function that returns the n-th Fibonacci number iteratively."
  "Explain the difference between TCP and UDP in one paragraph."
  "Write a one-line joke about pointers in C."
)

OK=0
for p in "${PROMPTS[@]}"; do
  body=$(printf '{"model":"%s","prompt":%s,"max_tokens":96,"temperature":0.0}' \
    "$MODEL" "$(printf '%s' "$p" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))')")
  http=$(curl -s -o /tmp/spec_gdn_resp.json -w "%{http_code}" \
    -X POST "http://127.0.0.1:$PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d "$body" --max-time 180)
  if [ "$http" = "200" ]; then
    OK=$((OK+1))
  fi
  echo "[capture] prompt http=$http (running total: $OK ok)" | tee -a "$SUMMARY"
done

echo "[capture] prompt suite done: $OK/12 200-OK" | tee -a "$SUMMARY"
COUNT=$(ls "$CAPTURES_DIR" 2>/dev/null | wc -l)
echo "[capture] tuples captured: $COUNT" | tee -a "$SUMMARY"
echo "[capture] flavor distribution:" | tee -a "$SUMMARY"
ls "$CAPTURES_DIR" 2>/dev/null \
  | sed -E 's/.*_(non_spec|spec_[A-Za-z0-9_]+)\.pt$/\1/' \
  | sort | uniq -c | tee -a "$SUMMARY"

echo "[capture] $(date -Iseconds) DONE" | tee -a "$SUMMARY"
exit 0

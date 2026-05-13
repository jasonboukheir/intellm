#!/usr/bin/env bash
# Drives `vllm bench serve` against a running server and parses the
# emitted JSON into the comparison table. Used for both the spec and
# non-spec runs (label distinguishes them).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/serve_common.sh"

LABEL="${1:?usage: $0 <label>}"
RESULT_DIR="${2:-/tmp/vllm-spec-decode/results}"
mkdir -p "$RESULT_DIR"

# Random dataset with fixed lengths: decode-heavy (long output) so the
# GDN spec-decode path is exercised. Single-stream concurrency keeps the
# verify shape inside the K=2 capture (size 3) instead of falling back
# to eager.
"$VLLM_BIN" bench serve \
  --backend openai-chat \
  --base-url "http://127.0.0.1:8000" \
  --endpoint /v1/chat/completions \
  --model palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4 \
  --served-model-name qwen3.6-35b-a3b \
  --dataset-name random \
  --random-input-len 256 \
  --random-output-len 512 \
  --random-prefix-len 0 \
  --num-prompts 20 \
  --num-warmups 2 \
  --max-concurrency 1 \
  --ignore-eos \
  --seed 42 \
  --label "$LABEL" \
  --save-result \
  --save-detailed \
  --result-dir "$RESULT_DIR" \
  --result-filename "${LABEL}.json" \
  --metadata "branch=$LABEL"

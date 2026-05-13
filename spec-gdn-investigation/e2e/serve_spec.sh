#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/serve_common.sh"

# qwen3_5_mtp: Qwen3.6 uses model_type=qwen3_5_moe (see speculative.py
# dispatch); the qwen3_next_mtp slot brutus had commented out is for the
# Qwen3-Next family. K=2 matches brutus's intended production setting
# and the cudagraph_capture_sizes=[1,4] sizing comment.
exec "$VLLM_BIN" "${COMMON_ARGS[@]}" \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":2}' \
  "$@"

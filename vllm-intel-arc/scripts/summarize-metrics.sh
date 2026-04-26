#!/usr/bin/env bash
# Summarize the engine stats + Prometheus delta from a profile-decode.sh run.
#
# Usage:
#   scripts/summarize-metrics.sh results/profile-<timestamp>/
set -euo pipefail

DIR="${1:-}"
if [ -z "$DIR" ] || [ ! -d "$DIR" ]; then
    echo "usage: $0 <results-dir>" >&2
    exit 1
fi

before="$DIR/metrics-before.txt"
after="$DIR/metrics-after.txt"
stats="$DIR/engine-stats.txt"
slog="$DIR/server.log"

echo "=== engine per-step stats (chronological tail) ==="
if [ -s "$stats" ]; then
    tail -40 "$stats"
else
    echo "(no engine-stats.txt — checking server.log directly)"
    grep -E "Avg prompt throughput|Avg generation throughput|Running:|GPU KV cache" "$slog" | tail -40 || true
fi
echo

# Prometheus diff for counters/gauges of interest
metric() {
    # metric NAME [LABEL_FILTER]
    local m="$1" filter="${2:-}"
    local re="^${m}(\{[^}]*\})? "
    if [ -n "$filter" ]; then
        re="^${m}\{[^}]*${filter}[^}]*\} "
    fi
    grep -E "$re" "$1file" 2>/dev/null | head -5
}

extract() {
    # Sum all values for a counter/gauge across labels
    # extract <file> <metric>
    awk -v m="$2" '
        $0 !~ /^#/ && $0 ~ "^"m"(\\{|\\b)" {
            # split off label and value
            split($0, parts, " ")
            v = parts[length(parts)]
            sum += v
        }
        END { printf "%.6f", (sum+0) }
    ' "$1"
}

if [ -s "$before" ] && [ -s "$after" ]; then
    echo "=== /metrics deltas (after - before) ==="
    for m in vllm:request_success_total \
             vllm:prompt_tokens_total \
             vllm:generation_tokens_total \
             vllm:request_prompt_tokens_sum \
             vllm:request_generation_tokens_sum \
             vllm:time_to_first_token_seconds_sum \
             vllm:time_per_output_token_seconds_sum \
             vllm:e2e_request_latency_seconds_sum \
             vllm:request_inference_time_seconds_sum \
             vllm:request_decode_time_seconds_sum \
             vllm:request_prefill_time_seconds_sum \
             vllm:request_queue_time_seconds_sum \
             vllm:cache_config_info \
             vllm:gpu_cache_usage_perc \
             vllm:num_requests_running \
             vllm:num_requests_waiting; do
        b=$(extract "$before" "$m")
        a=$(extract "$after"  "$m")
        d=$(awk "BEGIN { printf \"%.4f\", $a - $b }")
        echo "  $m   before=$b  after=$a  delta=$d"
    done
    echo

    # Headline numbers
    p_in=$(awk "BEGIN { printf \"%.0f\", $(extract "$after" vllm:prompt_tokens_total) - $(extract "$before" vllm:prompt_tokens_total) }")
    p_out=$(awk "BEGIN { printf \"%.0f\", $(extract "$after" vllm:generation_tokens_total) - $(extract "$before" vllm:generation_tokens_total) }")
    pre_t=$(awk "BEGIN { printf \"%.4f\", $(extract "$after" vllm:request_prefill_time_seconds_sum) - $(extract "$before" vllm:request_prefill_time_seconds_sum) }")
    dec_t=$(awk "BEGIN { printf \"%.4f\", $(extract "$after" vllm:request_decode_time_seconds_sum) - $(extract "$before" vllm:request_decode_time_seconds_sum) }")
    e2e_t=$(awk "BEGIN { printf \"%.4f\", $(extract "$after" vllm:e2e_request_latency_seconds_sum) - $(extract "$before" vllm:e2e_request_latency_seconds_sum) }")
    queue_t=$(awk "BEGIN { printf \"%.4f\", $(extract "$after" vllm:request_queue_time_seconds_sum) - $(extract "$before" vllm:request_queue_time_seconds_sum) }")
    ttft_t=$(awk "BEGIN { printf \"%.4f\", $(extract "$after" vllm:time_to_first_token_seconds_sum) - $(extract "$before" vllm:time_to_first_token_seconds_sum) }")

    echo "=== headlines ==="
    echo "  Prompt tokens processed:  $p_in"
    echo "  Output tokens generated:  $p_out"
    if [ "$pre_t" != "0.0000" ]; then
        echo "  Prefill cumulative time:  ${pre_t}s   (=> $(awk "BEGIN{printf \"%.1f\", $p_in / $pre_t}") prompt-tok/s aggregate)"
    fi
    if [ "$dec_t" != "0.0000" ]; then
        echo "  Decode cumulative time:   ${dec_t}s   (=> $(awk "BEGIN{printf \"%.1f\", $p_out / $dec_t}") gen-tok/s aggregate)"
    fi
    if [ "$ttft_t" != "0.0000" ]; then
        echo "  TTFT cumulative time:     ${ttft_t}s"
    fi
    echo "  Queue cumulative time:    ${queue_t}s"
    echo "  E2E cumulative time:      ${e2e_t}s"
fi

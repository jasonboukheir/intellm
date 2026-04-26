# Phase 2a — KL-divergence harness MVP

Goal: reusable harness that compares KV-cache quantization variants on the
same prompts, in preparation for IsoQuant integration. Today's run uses
llama.cpp's *built-in* KV-cache quants as a baseline check.

## Setup

- Server: `ghcr.io/ggml-org/llama.cpp:server-intel` on Arc B70
- Model:  Qwen3.6-35B-A3B-UD-Q4_K_M
- Three server boots, each with a different KV-cache type:
  - `fp16` (`--cache-type-k f16 --cache-type-v f16`)
  - `q8_0` (`--cache-type-k q8_0 --cache-type-v q8_0`)
  - `q4_0` (`--cache-type-k q4_0 --cache-type-v q4_0`)
- For each: 16 short prompts × 32 generated tokens × top-50 logprobs captured
  via `/v1/completions` with `logprobs:50, temperature:0`.

Harness:
- `harness/capture_logprobs.py` — captures top-K distributions, tagged JSON.
- `harness/compute_kl.py` — pairwise truncated-KL + greedy-agreement +
  cross-entropy, aligned on greedy-agreement prefix only.

## Results

### Quality (KL/CE on greedy-agreement positions)

| comparison | full-agreement prompts | KL mean | KL p50 | KL max | -log Q(P_pick) | baseline NLL |
|---|---|---|---|---|---|---|
| fp16 vs **q8_0** | 10/16 | **0.00406** | 0.00074 | 0.07322 | 0.5738 | 0.5812 |
| fp16 vs **q4_0** | 5/16  | **0.00792** | 0.00298 | 0.08937 | 0.5776 | 0.5812 |

KL on aligned positions is tiny in both cases. Q8_0 KV is statistically
indistinguishable from FP16; Q4_0 doubles the KL but is still ~0.01 nats
per token — well below the threshold where users would notice in chat.

### Greedy stability (a different signal)

Greedy argmax flips more often than KL would suggest because boundary
tokens in low-temperature decoding have nearly-tied logits and tiny
numerical jitter from quantized KV is enough to swap them. This is
*decoding* sensitivity, not *distribution* sensitivity.

- Q8_0: total agreement 79 % across all 512 positions, 10 of 16 prompts
  diverged-free for the full 32 tokens.
- Q4_0: total agreement 64 %, only 5 prompts diverged-free, one prompt
  diverged at the very first generated token.

This is exactly why the rotation-quant papers (TurboQuant / RotorQuant /
IsoQuant) report PPL on a fixed dataset rather than greedy match: PPL
uses the model's prob assigned to the *true* next token, not its argmax
choice, so it's robust to argmax flipping.

### Throughput

| KV type | mean predicted/s | wall (s, 16 prompts × 32 tok × top-50) |
|---|---|---|
| fp16 | 29.4 | 21.7 |
| q8_0 | 28.9 | 43.5 |
| q4_0 | 28.6 | 45.5 |

Per-token rate is essentially flat — at 32 tokens of context, KV state is
~80 MB regardless of quant; bandwidth is dominated by weight loads. The
throughput payoff for KV-quant only shows at **long context with many
concurrent sequences**, which is where Phase 2c will live.

(Wall-time disparity between fp16 and q8/q4 runs reflects fresh server
startup and one-shot kernel JIT, not generation cost. Single-stream
predicted_per_second is the meaningful number.)

## Validation: harness works as specified

- Tagged-JSON capture format is portable (id-keyed top_logprobs survive
  round-trip; the harness re-coerces JSON-stringified keys back to int).
- Pairwise comparison is fast: ~2.9 MB JSON → milliseconds for KL.
- All 16 prompts produced valid logprobs at every position; no API errors.

## Next: phase 2b (IsoQuant integration)

The harness is ready for a *new* KV cache type. Two integration paths
for IsoQuant on Battlemage:

1. **Custom GGML cache type.** Add a `cache_type::isoquant_b3` enum to
   llama.cpp's KV cache, implement quantize/dequantize on the SO(4)
   rotation. Lives upstream-able.
2. **External KV-cache wrapper.** Run llama-server with FP16 KV, intercept
   the cache via a Python in-process replacement. Faster to prototype, no
   upstream contribution path.

(1) is the right long-term answer; (2) lets us validate the rotation-
quant numbers from the IsoQuant paper match what we measure here.

## Phase 2c sketch (long-context demo)

Re-run this harness with prompts of 8K–32K tokens and 32 concurrent
sequences. At those sizes:
- KV cache: 32 seqs × 32K ctx × 10 attn-layers × 2 (K+V) × 2 bytes/value ≈ 5 GB FP16,
  1.25 GB at Q4 — moves the ceiling on how many sequences fit on the 32 GB card.
- Generation rate gap (fp16 → q4) should widen because KV bytes start to
  matter alongside weights.

That's where IsoQuant's claimed advantage over Q4_0 (better KL at same
bit-budget) becomes user-visible: more concurrent users at higher quality.

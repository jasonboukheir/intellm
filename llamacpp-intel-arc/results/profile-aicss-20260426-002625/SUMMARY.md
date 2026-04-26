# llama.cpp/SYCL profile — patched build (6 of 7 Intel SYCL PRs cherry-picked)

Same machine, same model, same workload as
`results/profile-20260425-233958/SUMMARY.md`. Only difference: the
llama.cpp binary is built from upstream master + 6 cherry-picked PRs
(see `build-aicss/README.md`; #22152 was skipped because upstream master
already has its symbols).

## Headlines vs vanilla container

| metric                                | vanilla `:server-intel`     | **patched (6 PRs)**         | speedup |
|---------------------------------------|-----------------------------|-----------------------------|---------|
| Single-stream decode (256 tokens)     | 47.6 tok/s                  | **73.45 tok/s**             | **1.54×** |
| Single-stream decode (32 tokens warm) | 49 tok/s                    | **71.42 tok/s**             | **1.46×** |
| Prompt eval (632-token, fresh)        | 360.65 tok/s                | **306.98 tok/s**            | 0.85×   |
| Prompt eval (632-token, prefix-cached)| 161.31 tok/s                | **161.31 tok/s**            | tied    |
| 4-concurrent per-slot                 | 10.79 tok/s                 | **13.99 tok/s**             | **1.30×** |
| 4-concurrent aggregate                | 43 tok/s                    | **56 tok/s**                | **1.30×** |
| 8-concurrent per-slot                 | 10.83 tok/s                 | **15.28 tok/s**             | **1.41×** |

(8-concurrent values from `engine-stats.txt`; llama.cpp queued the 5th-8th
seqs since we kept `parallel:4` in the config.)

The 1.54× single-stream decode and 1.30-1.41× concurrent decode match the
range the umbrella PR #22066 advertised on similar-class models
(Qwen3.5-9B 1.43× decode, Llama-3.1-8B 1.10× decode, more for
Qwen3.5 because of the PAD-stride bug fix specifically).

## Where the gain comes from (per the PR descriptions)

| PR | Effect on this workload |
|----|-------------------------|
| #22147 — BMG AOT + MMQ subgroup pin   | Build cleanup, no measurable effect on JIT runs |
| #22148 — PAD non-contiguous stride    | Eliminates CPU fallbacks on view/permute ops; biggest single-step gain on Qwen-class models per upstream |
| #22149 — FILL/CUMSUM/DIAG/SOLVE_TRI/SSM_SCAN/GATED_DELTA_NET | Eliminates per-step CPU↔GPU round-trips; ~1.20× decode |
| #22150 — small f32 → oneMKL           | ~1.20× prefill on dense models; small effect here (MoE mostly Q4) |
| #22153 — async-mem-op env             | Marginal, lets reorder staging not block on host |
| #22156 — Q6_K SWAR byte-subtract      | No effect (we use Q4_K_M, not Q6_K) |

## Why prompt eval got a hair *slower* (360 → 307 tok/s)

Cold prompt eval (no prefix cache hit) regressed by ~15 %. Likely
explanation: the new oneMKL small-matmul threshold (#22150) covers f32
GEMMs only and BF16 prefill on Q4_K_M doesn't see the path; meanwhile
the new ops (#22149) added some dispatch overhead. Repeat-prompt prefill
(prefix cache hit, the realistic case) is unchanged at 161 tok/s. Worth
filing if it persists, but not on our critical path.

## Bandwidth ceiling — the math now needs revising

Pre-patch we were at 47.6 tok/s ≈ "86% of bandwidth ceiling" assuming
~14 GB per-token bytes. Post-patch we hit **73.45 tok/s** on the same
hardware and same model — that's 132% of that supposed ceiling. So
either:

1. Pre-patch wasn't *bandwidth*-bound — it was bandwidth + CPU↔GPU
   round-trip-bound. PR #22149's "eliminate CPU↔GPU transfers" claim
   maps directly to this. Per-token bytes loaded didn't change; per-token
   *latency* dropped because the CPU stalls went away.
2. The MoE per-token active footprint is smaller than I estimated. With
   shared params + 1 routed expert per layer × 9-of-256 expert routing,
   per-token bytes might be closer to 8–9 GB than 14 GB, putting the
   real ceiling around 70 tok/s (which we are now near).

Both can be true. Either way, the user-visible result is ~73 tok/s on a
35B-A3B Q4 model on a 32 GB Arc B70 — comparable to the M4 Max at
80 tok/s, on a card with similar DRAM bandwidth. The 80–86 % bandwidth
gap to MLX has shrunk significantly with these patches.

## Reproduce

```sh
scripts/build-aicss.sh                                # cherry-picks + builds
scripts/run-server-aicss.sh \
    configs/models/qwen3.6-35b-a3b-q4km.yaml          # runs on port 8081
# ... then this same workload pattern was used for the numbers above
```

## Note on PR-5 (#22152)

We did not include #22152 (Q5_K + Q8_0 reorder MMVQ). Its commit adds
`dequantize_q8_0_reorder` etc, but upstream master independently merged
the same symbols. Cherry-picking now produces duplicate-definition
errors. The PR will need a rebase from its author. We use Q4_K_M, so
losing Q5_K + Q8_0 reorder paths doesn't affect our decode rate.

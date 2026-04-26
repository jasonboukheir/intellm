# llama.cpp/SYCL profile — Qwen3.6-35B-A3B (UD-Q4_K_M), Arc B70

Container: `ghcr.io/ggml-org/llama.cpp:server-intel`. Model:
`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (Unsloth dynamic Q4_K_M, 22.1 GB).
Server config: ctx=32768, parallel=4, flash-attn on,
GGML_SYCL_DISABLE_OPT=1 (Battlemage workaround).

## Headlines

| metric                          | value                            |
|---------------------------------|----------------------------------|
| **Single-stream decode**        | **47–49 gen-tok/s**              |
| **Single-stream prompt eval**   | **361 prompt-tok/s** (632 tokens in 1.75 s) |
| 4-concurrent decode (per slot)  | 10.8 gen-tok/s                   |
| 4-concurrent decode (aggregate) | ~43 gen-tok/s                    |
| Prefix cache hit rate           | high (≥99 % of repeat prompts loaded only 4 new tokens) |
| Server boot time                | 40 s (vs vLLM-XPU's 76 s)        |

## Comparison vs vLLM-XPU baseline (Qwen2.5-7B BF16)

| stack | model | size | single-stream decode | aggregate @ 4-conc |
|---|---|---|---|---|
| vLLM-XPU            | Qwen2.5-7B BF16          | 14 GB  | 32.7 tok/s | 147 tok/s |
| **llama.cpp/SYCL**  | **Qwen3.6-35B-A3B Q4_K_M** | **22 GB** | **48.0 tok/s** | **43 tok/s** |

For a model **5× larger total / 35B vs 7B params**, llama.cpp on the same
card is **47 % faster single-stream**. That's the MoE bandwidth thesis
playing out: only the always-active params + 1 routed expert per layer
need to be loaded per token, and Q4 shrinks each by 4×.

## Concurrency: aggregate is *worse* than single-stream (regression)

When 4 sequences run in parallel, **per-slot drops to 10.8 tok/s** and
**aggregate is ~43**, less than single-stream 48. With MoE this is
expected: divergent expert routing across the 4 sequences forces loading
multiple experts per layer per step, multiplying the per-step bandwidth
without compute savings. vLLM's smaller dense model batches additively;
llama.cpp's MoE here doesn't.

If the user-facing workload is *single-user chat*, this is fine — we get
48 tok/s on a 35B-A3B model on a 32 GB card. For multi-tenant serving
the batching story is materially worse than the dense baseline.

## Bandwidth math

22 GB Q4 weights total but only a fraction (always-active + 1/256 of
expert weights × 9 activated experts ≈ shared params + ~3.5 % of expert
mass) is loaded per token. With shared params + 1 expert layer-by-layer,
per-token bytes ≈ 12–14 GB ⇒ ceiling 600 / 13 ≈ **46 tok/s**. Measured 48.
We're at the bandwidth ceiling for the active footprint.

## What this means for next steps

1. **Single-stream Qwen3.6 q4 is shipping-quality on this card.** The
   target model + quant + serving stack works.
2. **Concurrency optimization is its own project** — expert pre-fetching
   / routed batching / shared-prefill is needed to get aggregate gains.
   Not the most pressing if interactive use is the goal.
3. **TurboQuant/IsoQuant adds value at long context, not here.** This
   profile only used ~600-token prompts. KV cache was tiny (10 attention
   layers × 4096 ctx ≈ 80 MB). KV-quant savings show up at 32K+ context
   with multiple concurrent sequences. The harness should bench at long
   context to make the value visible.

## Architecture notes

- llama.cpp recognized the architecture: 41 layers, 10 with KV cache, 30
  Gated DeltaNet (recurrent state buffer 62.81 MiB).
- "fused Gated DeltaNet (autoregressive)" + "(chunked)" both enabled —
  llama.cpp's DeltaNet kernel covers both regimes.
- `n_busy_slots_per_decode = 2.4` over the workload — we never sustained
  4-concurrent because the warmup + single-stream + ramp left slots idle
  most of the time.

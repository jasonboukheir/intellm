# vLLM workload profile — Qwen2.5-7B-Instruct, Arc B70 (Battlemage)

Container: `intel/vllm:0.17.0-xpu` (vllm v0.1.dev14456). BF16 weights,
chunked prefill enabled, max_model_len=8192.

## Workload

14 requests across 3 concurrencies (1, 4, 8), prompts ~600 tokens, outputs
32–256 tokens. Mix biased toward decode.

## Headlines

| metric | value |
|---|---|
| Prefill cumulative time | **1.18 s** for 8,848 prompt tokens (7,528 prompt-tok/s) |
| Decode cumulative time  | **35.66 s** for 1,312 generated tokens (36.8 gen-tok/s per request) |
| Queue time              | ~0 s (no scheduler contention at this load) |
| KV cache utilisation    | 0.5 % of 224,192-token capacity |
| **Time split**          | **prefill 3.2 % / decode 96.7 %** |

## Bandwidth reality check

- Qwen2.5-7B BF16 ≈ 14 GB on disk and in VRAM.
- Decode loads the entire model **per token**.
- Battlemage Arc B70 peak DRAM bandwidth: ~600 GB/s.
- Theoretical decode ceiling: 600 / 14 ≈ **42.9 tok/s** single-stream.
- Measured: **~37 tok/s ⇒ 86 % of physics**.

Decode is memory-bandwidth bound. Attention / matmul compute on XMX has
plenty of slack — but it doesn't matter while we're flat against DRAM.

## What does and doesn't move the needle on Battlemage

| lever                              | helps decode? | notes |
|------------------------------------|---------------|-------|
| Custom flash-attention kernel      | **NO**        | vLLM-XPU already uses FA2; attention isn't on the critical path here |
| KV-cache quantization (TurboQuant) | weakly        | KV is 0.5 % of bytes loaded at our context lengths; only matters at long context |
| Weight quantization (Q4 / MXFP4)   | **YES**       | 4× fewer bytes/token ⇒ ~4× decode ceiling |
| MoE expert sparsity                | **YES** for MoE | Per-token bandwidth drops to active-expert footprint |
| Speculative decoding               | **YES**       | Multiple tokens per weight-load |
| Tensor parallelism (more GPUs)     | **YES**       | Doubles aggregate BW |

## Implication for Qwen3.6-35B-A3B target

35B total / 3B active MoE + hybrid DeltaNet/Attention. With Q4 weights
(~17.5 GB total) the model fits the 32 GB card. With MoE-aware loading
(skip unrouted experts per token) effective bytes-per-token drops to
3–5 GB ⇒ decode ceiling 120–200 tok/s.

Blocker: vLLM-XPU does not load GGUF, AWQ, GPTQ, or BitsAndBytes (only
FP16, dynamic FP8, MXFP4 per docs). Unsloth ships GGUF + MLX. Two paths:

1. **llama.cpp with the SYCL backend** — runs Unsloth GGUF directly.
   No infrastructure to build. Loses vLLM scheduler / OpenAI compat.
2. **MXFP4 conversion of Qwen3.6** for vLLM-XPU. More plumbing; keeps
   the existing serving stack.

Both still leave the 86%-of-bandwidth-ceiling property in place; the win
is reducing bytes-per-token, not improving compute utilization.

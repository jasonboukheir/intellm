# Spec-decoding (GDN) e2e perf — Qwen3.6-35B-A3B-GPTQ-Int4

Closes the open rung 10 from `spec-gdn-investigation/progress/README.md`:
*"e2e Qwen3.6+MTP-K3 with acceptance-rate benchmarking"* (run with K=2 to
match the production cudagraph capture sizes that brutus's
vllm-xpu-chat unit was tuned for).

## Setup

- Host: brutus (Intel Arc B70, 32 GiB VRAM)
- vllm: `0.20.2.dev0+xpu.unstable` from
  `/nix/store/n468660n…-python3.12-vllm-xpu-0.20.2.dev0+xpu.unstable`
  (cache-hit of intellm `.#vllm-xpu-unstable.withTorchvision true` after
  bumping `vllm-xpu-nix` 0.df27cb5 → 0.9fb4a64 to match brutus).
- vllm src @ `ee8e4c3` (`xpu: spec-decode-aware GDN attention dispatcher`)
- vllm-xpu-kernels src @ `99e9a4a` (`xpu: spec-decoding-aware GDN
  attention kernel` — tick-40 FLA-aligned conv1d fix)
- Model: `palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4` @ d1fef18
- Quantization: INC (GPTQ-sym-int4 MoE → `xpu_fused_moe(is_int4=True)`)
- KV: `turboquant_k3v4_nc` (3-bit MSE-Lloyd-Max K + 4-bit V)
- max_model_len 65536, max_num_seqs 32, gpu_memory_utilization 0.83
- cudagraph_capture_sizes `[1, 4]` (rounds to `[3, 12]` under K=2)

## Bench

`vllm bench serve` with `--dataset-name random --random-input-len 256
--random-output-len 512 --num-prompts 20 --num-warmups 2
--max-concurrency 1 --ignore-eos --seed 42`, OpenAI-chat backend so the
Qwen3.6 chat template + reasoning prefix go through the real serving
path.

## Results

| metric | baseline | spec (MTP K=2) | delta |
|---|---|---|---|
| output throughput (tok/s) | **58.85** | **75.80** | **+28.8%** (1.288×) |
| wall-clock duration (s) | 173.99 | 135.09 | −22.4% |
| median TPOT (ms) | 16.92 | 13.08 | −22.7% |
| median TTFT (ms) | 94.61 | 115.70 | +21.1 ms (MTP draft setup) |
| median ITL (ms) | 16.72 | 29.71 | +12.99 ms (1+1.29 tokens/emit) |
| acceptance rate | — | 64.46% | — |
| mean acceptance length | — | 2.29 / 3 | — |
| per-position acceptance | — | P0 74.68%, P1 54.25% | — |

20 requests, 256-token prompts, 512-token outputs, single-stream
verify (max_concurrency=1).

## Interpretation

- **Functional rung 10 GREEN.** The spec-decode-aware GDN dispatcher
  + tick-40 FLA-aligned conv1d kernel ran clean for all 20 requests
  with no failures and no per-step fallback to FLA. The kernel
  emits state updates compatible with the K=2 MTP verify path on the
  linear-attention layers (30 of 40 are `linear_attention`, the rest
  `full_attention`).
- **28.8% decode throughput gain at K=2.** Matches what a 64%
  acceptance rate with `acceptance_length=2.29` predicts: each spec
  step verifies ~2.3 tokens for the cost of the target forward plus
  the MTP draft (per `llm_base_proposer.py` re-running the MTP layer
  K times under `num_speculative_tokens=2`, the documented cost we
  knew going in from the speculative.py:672 warning).
- **TTFT goes up ~21 ms** with spec on; this is the one-time MTP
  draft graph dispatch ahead of the first verify. Expected and small
  vs. the wall-clock win on long outputs.
- **ITL is 29.71 ms** under spec vs. 16.72 ms baseline because each
  emit produces 1 + accepted tokens (mean 2.29). Wall-clock per
  *generated token* is what matters → TPOT is the right comparator
  (13.08 vs. 16.92 ms, −22.7%).

## Notes worth filing (none are vllm-xpu-kernels bugs)

- `WARNING [speculative.py:525] method qwen3_5_mtp is deprecated and
  replaced with mtp.` Internal alias swap; both names work. Suggests
  updating brutus's commented-out config from `qwen3_next_mtp` to the
  generic `mtp` rather than `qwen3_5_mtp` (or just leave `qwen3_5_mtp`
  for clarity; the dispatcher routes correctly).
- `Triton kernel JIT compilation during inference` warnings for 11
  kernels (e.g. `eagle_prepare_next_token_padded_kernel`,
  `_tq_decode_stage1`, `_causal_conv1d_update_kernel`,
  `rejection_greedy_sample_kernel`). These are warmup-coverage gaps in
  vLLM's spec-decode init path on XPU, not kernel bugs — the spike
  is one-time per shape.
- `WARNING [arg_utils.py:2035] TurboQuant is not yet compatible with
  FlashAttention >= 3.` Auto-downgrade to FA2 is the documented
  behavior, mentioned just for the bench log.

Raw JSON: `/tmp/vllm-spec-decode/results/{spec,baseline}.json`
Server logs: `/tmp/vllm-spec-decode/logs/{spec,baseline}.log`

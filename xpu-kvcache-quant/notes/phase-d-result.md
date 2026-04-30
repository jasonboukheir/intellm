# Phase D result — 2026-04-30

Outcome: **TurboQuant + Qwen3.6 hybrid MoE works end-to-end.** KL stays within paper bounds; quality cost from disabled boundary protection is real but bounded.

Model: `Qwen/Qwen3.6-35B-A3B` (35B BF16, MoE, hybrid Gated DeltaNet + full attention, 64 layers / 16 full-attn / 48 linear-attn). Image: `vllm-xpu-tq:hybrid-noskip-4088d3dd0` (vllm `0.20.1rc1.dev87+g4088d3dd0`). Single Arc B70 (32 GB), `--cpu-offload-gb 55 --max-model-len 1024 --gpu-memory-utilization 0.78 --enforce-eager --max-num-seqs 1`.

## Patch chain that made this work (4 commits on `jasonboukheir/vllm` `tq-hybrid-allow`)

1. `1207f07d5` — `arg_utils.py`: boundary skip uses full-attention indices (not `range(num_layers)`) for hybrid models.
2. `658927c23` — `platforms/interface.py:_align_hybrid_block_size`: TQ-aware per-token bytes via `TQFullAttentionSpec(tq_slot_size=...)`.
3. `9ad5531f2` — `kv_cache_utils.py:unify_kv_cache_spec_page_size`: pads via `page_size_padded` when no integer ratio.
4. `4088d3dd0` — `arg_utils.py`: hybrid branch sets `boundary = []` (workaround until reshape generalizes).

Diagnostic chain in `phase-a5-result.md`. The big remaining blocker was patch (4): `_reshape_kv_cache_tensors` (`gpu_model_runner.py:6644`) can't view padded mixed-page layouts when `num_blocks_per_kv_block > 1` (non-MLA backends), so we suppress the BF16 boundary-skip layers entirely on hybrid.

## Short-context KL: TQ k8v4 vs BF16 baseline

```
prompts:           16 × 64 tokens (kl_test_set.txt)
greedy agreement:  33.7%   (worst-prompt 3.1%, full-agreement 2/16)
KL(P||Q) per agreed-tok:  mean=0.01262   p50=0.00583   max=0.0872
ce_q minus ce_p:   +0.183 nats          (≈ +20% PPL inflation)
```

Comparison to Phase C uniform Qwen3-4B + boundary protection:
| metric | Phase C uniform Qwen3-4B (with boundary skips) | Phase D hybrid Qwen3.6 (no boundary skips) |
|---|---|---|
| greedy agreement | 26.4% | **33.7%** |
| KL mean | 0.00752 | **0.01262** |
| ce_q − ce_p | +0.042 | **+0.183** |

Reading: still within the paper's k8v4 envelope (0.001-0.05 nats), so the kernel path is sound. But `ce_q-ce_p` is ~4× the uniform-attn baseline — boundary protection was doing real work, and disabling it on hybrid is paying ~20% PPL inflation. **Patch-5 (reshape generalization to re-enable boundary protection on hybrid) is now a quality fix, not a correctness completion.**

## Throughput

- Cold first-token: ~20s for 8 tokens (prefill + first-token cache miss)
- Sustained sequential: **~6.8 tok/s** (16×64 tokens in 150.6s) vs the 0.4 tok/s warned in CONTINUE.md (that figure included prefill in the divisor)
- Available KV pool: 180,736 tokens / max concurrency 176.5×

## What's filed as memory

- `project_tq_hybrid_patches.md` — patches 1-4 with reshape-pipeline gap details
- `project_sym_int4_forward_port.md` — sym_int4 forward-port (commits `1aeec4306` `1db471bde` `2756dd749` `db67f73b0`); STRANDED — IPEX is EOL March 2026 (issue #867). Don't extend.
- `project_xpu_int4_moe_path.md` — vllm-xpu-kernels W4A16 MoE path; `XPUExpertsWNA16` foundation committed (`38237e347`); WNA16 oracle XPU branch + INC MoE method + checkpoint adapter still TODO.
- `project_vllm_turboquant_pr38479.md` — upstream backend reference

## Captures

- `xpu-kvcache-quant/results/phase-d/qwen36_short_bf16.json` — BF16 baseline, 16 prompts × 64 tokens
- `xpu-kvcache-quant/results/phase-d/qwen36_short_k8v4.json` — TQ k8v4 same prompts
- `xpu-kvcache-quant/results/phase-d/kl_short_k8v4.json` — KL output

## Status of subtasks

- [x] Patch chain to TQ + hybrid (4 commits)
- [x] Boot Qwen3.6-35B-A3B + TQ k8v4 end-to-end on B70
- [x] Generation smoke test (coherent decode)
- [x] BF16 baseline capture
- [x] TQ k8v4 capture
- [x] KL compute (this document)
- [x] sym_int4 forward-port (code-complete; runtime-blocked)
- [x] Docker image recipe (code-complete; runtime-blocked)
- [x] `XPUExpertsWNA16` foundation
- [ ] WNA16 oracle XPU backend branch
- [ ] `INCXPUMoEMethod` wrapping `XPUExpertsWNA16`
- [ ] GPTQ-Int4 → `xpu_fused_moe(is_int4=True)` layout adapter
- [ ] Long-context (1k, 4k) KL captures
- [ ] Patch 5 (reshape generalization) to re-enable boundary protection on hybrid

Phase D ✅ done. Strategy pivoted at end of session — IPEX path retired, vllm-xpu-kernels W4A16 path is the upstream-aligned future. See CONTINUE.md for next session.

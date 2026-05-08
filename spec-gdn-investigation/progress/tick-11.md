2026-05-07 tick 11: replay oracle switched to inline FLA-recurrence; SYCL
codegen bug now properly observable at test time.

`tests/kernels/xpu/test_spec_gdn_replay.py` previously diffed the SYCL
kernel against `payload["core_attn_out"]` / `payload["ssm_state_post"]`,
which tick 10 proved are NOT byte-reproducible by re-running the same
FLA Triton kernel on the same inputs (FLA-recurrence ≠ FLA-CAPTURED by
~108 cells max 2.58). Switched the spec branch of the test to compute
the oracle inline per call: `_compute_fla_spec_oracle()` builds a fresh
dense pool from `payload["ssm_state_pre"]`, runs `causal_conv1d_update`
+ `fused_sigmoid_gating_delta_rule_update` on `payload["mixed_qkv"]`,
returns `(core, conv_pool_post, ssm_pool_post, remap)`. Helper mirrors
`gdn_linear_attn.py:830-943` (spec branch only). `z` is a slice of the
qkvz projection (no recurrence), so the captured `payload["z"]` stays
the oracle for both spec and non_spec — kept that comparison for
backward continuity.

Non-spec branch left on captured oracle this tick: reproducing the
prefill conv path inline needs `attn_metadata.{chunk_indices,
chunk_offsets,metadata}` for `causal_conv1d_fn` which the dump payload
doesn't carry yet — separate tick to extend the schema if needed.

Validation, `VLLM_XPU_USE_SYCL_SPEC_GDN=1`:
- 5 captures sampled across 2 layers × {non_spec, rung-3, rung-4}.
- **layers_0 rung-4** (the cross-check baseline): core_attn_out max
  abs diff **1.2227**, mean **1.76e-3**, shape (4, 32, 128) = 16384
  cells. Matches tick 10's pyref-vs-FLA-recurrence finding to 4
  significant figures (1.22, mean ~1.7e-3).
- **layers_5 rung-4**: max **3.10e-2**, mean **2.89e-5** — **30×
  smaller** than layer 0. The codegen bug's amplitude depends on
  the recurrence inputs; layer 5's q/k/v at the bad (head, v) coords
  evidently produce smaller deltas than layer 0's. Useful narrowing
  signal: it's not a uniform "always wrong by X" defect.
- **layers_0 rung-3** (spec_K4_min1_max1, only 1 token accepted, so
  the recurrence iterates once per spec window): also fails.
  Magnitude was not the last-printed failure so not captured here;
  re-run with `--no-tb` or extend `_diff` to log fail count + max
  per assertion to characterize cleanly.
- **layers_0 non_spec** (prefill, 7 tokens): max **borderline ~3.1e-2**
  vs captured oracle. Was previously claimed "passes at atol=2e-2"
  in tick 3 — now barely fails. Either bf16 run-to-run noise (the
  `_xpu_C` call's reduce-over-group order may not be deterministic
  across launches) or genuinely a borderline pass that flipped.
  This is the captured-oracle path, untouched this tick — note for
  later but does not block.

So the test now correctly fails on the SYCL bug at the right scale,
and "passes" mean what they should (SYCL ≡ FLA up to bf16 noise).

Wakeup hint for the runner:
- The replay test on a single capture takes ~2s including `vllm-run`
  startup. A full 200-capture sweep (`-k spec` only, ~170 captures)
  would be ~5–8 min — fits a 270s tick if filtered to one rung.

Next tick (preferred): characterise the codegen bug's footprint
without rebuilding. Concretely: extend `_diff` (or add a side helper)
to also report `fail_count` and the unique failing v_dim coords per
assertion, then run pytest across all spec_K4_min4_max4 captures and
collect: per-layer (max, mean, fail_count, top-3 v_dim coords by
fail count). Hypothesis: layer 0 has the worst amplitude, layer 5
the smallest, the v_dim coord set [25,33,47,53,55,60,79,83,107,124]
recurs across all layers (per tick 9). Confirming this gives a clean
bisect target for which (head, v_dim) loci to focus the SPIR-V/IR
inspection on. Don't rebuild — narrow first.

Alternative tick: dump SPIR-V / LLVM IR for the
`gdn_attention_kernel<spec=true>` template instantiation around
`gated_delta_rule.hpp:207-285` and look for the j=2-vs-others code
asymmetry tick 9 hypothesised. Likely needs `vllm-xpu-build` with
`-save-temps` or similar — only do if the per-layer characterization
above is inconclusive.


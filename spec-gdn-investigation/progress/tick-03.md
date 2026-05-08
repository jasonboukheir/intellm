2026-05-07 tick 3: Triton-side blockers + harness bug, captures landed,
first SYCL-vs-FLA diff result.

**Triton blockers** (Intel backend, Triton 3.7):
- `TritonIntelStrideVersioning` MLIR pass crashes on
  `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` (FLA prefill) for
  every BV×num_warps×num_stages config. Pass is added unconditionally
  in `triton/backends/intel/compiler.py:make_ttir`; no env knob.
- `Autotuner._bench` only catches OutOfResources /
  CompileTimeAssertionFailure / PTXASError; the generic
  `RuntimeError("PassManager::run failed")` bypassed it and killed
  the EngineCore on the very first request. With the autotuner-only
  fix, autotune finished but the picked "winner" still crashed at
  post-autotune compile.
- Workaround: site-init `.pth` patch that (a) no-ops
  `intel.passes.ttir.add_stride_versioning` (correctness-safe — it's
  an opt pass) and (b) broadens `_bench`'s catch to absorb
  MLIR-pass `RuntimeError`s. Source of truth:
  `vllm/.spec-gdn-triton-autotune-patch.py`. Staged in container as
  `/opt/venv/lib/python3.12/site-packages/_spec_gdn_triton_patch.{py,pth}`
  (lost on `vllm-xpu-clean`; restage from the source-of-truth file).

**Capture run** (`vllm/.spec-gdn-capture.sh`, 12 prompts post-/health):
12/12 200-OK, 200 tuples in `/tmp/spec_gdn_captures` (inside container).
Distribution: 30 `non_spec`, 90 `spec_K4_min1_max1`, 80 `spec_K4_min4_max4`.
Note: K=4 (= num_speculative_tokens+1 = 3+1), not K=3 as the
original ladder language implied. Acceptance flavors are at the
extremes (1-of-4 and 4-of-4); no mid-acceptance captures yet.

**Replay harness bug**: `_build_dense_pool` did
`conv_pre[keep].to(device)` with `keep` already on device, throwing
"indices should be either on cpu or on the same device as the indexed
tensor". Fixed by moving `conv_pre`/`ssm_pre` to device once up-front
and indexing `slots.to(remap.device)`. (Test had never been run on
real captures before.)

**First SYCL-vs-FLA diff** (`tuple_..._layers_0_..._non_spec.pt`,
num_actual=7, prefill, single slot):
- `core_attn_out`, `z`, `conv_state_post` all pass at atol=rtol=2e-2.
- `ssm_state_post`: **4 / 524288 cells exceed tolerance** (0.0008%),
  concentrated:
  - 3 cells at `(slot=0, head=14, k=79, v∈{21,51,82})`,
    deltas 0.040–0.080 on values ~0.3–0.93 (6–19% relative)
  - 1 cell at `(slot=0, head=21, k=107, v=4)`, delta 0.035 on 0.37
    (10% relative)
- No NaN/Inf, mean abs diff 2.5e-5 (rest is bf16 noise floor).
- Token outputs are byte-correct vs FLA — divergence is purely in
  the SSM-state writeback at very specific (head, k) coordinates.
- This is `non_spec` — pre-existing baseline divergence, not from
  the new spec branch.

Narrow tooling staged: `vllm/.spec-gdn-narrow.py` (loads one tuple,
runs `_call_sycl`, prints the failing cell list).

Next blocker: investigate why `ssm_state_post` diverges at
`(head=14, k=79)` and `(head=21, k=107)` — tile/warp boundary in
the SYCL native kernel? Compare against gated_delta_rule.hpp's
state-write path for those K-coord ranges.


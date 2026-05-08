2026-05-08 tick 46: **Rungs 7 & 8 GREEN — harness extended to
synthesise the missing coverage**.

Tick 45 closed the per-tick loop with rungs 7 (mixed batch) and 8
(gqa_interleaved=True) marked deferred because the captures in
`/tmp/spec_gdn_captures` are all single-flavor and all
`gqa_interleaved_layout=False`. The kernel-side correctness was
confirmed; only the harness needed to grow.

This tick wires that coverage into
`tests/kernels/xpu/test_spec_gdn_replay.py` without taking a new
capture run.

## Rung 7 — synthetic mixed batch

`test_sycl_mixed_batch_matches_per_subset` pairs each non_spec
capture with the same-layer `spec_K4_min1_max1_000001` capture
and synthesises a mixed batch:

- `mixed_qkvz = cat([ns.qkvz, sp.qkvz], dim=0)` (and ba)
- `non_spec_token_indx = arange(0, n_ns)`
- `spec_token_indx = arange(n_ns, n_ns + n_sp)`
- Unified pool spans both captures' slots in disjoint dense ranges
  (1..n_ns_slots are non_spec, then spec slots after).
- The test replicates the mixed-batch branch of
  `_gdn_xpu_spec_sycl_path:604-676`: split via `index_select`, two
  `gdn_attention` calls (one with non_spec args, one with the
  spec_state_indices_tensor + num_accepted_tokens kwargs),
  scatter back via `index_copy_`.
- Output buffers are pre-filled with NaN so a missing scatter is
  loud rather than near-matching.
- Each subset is diffed against its own reference (non_spec ↔
  captured payload outputs, spec ↔ inline FLA oracle from
  `_compute_fla_spec_oracle`).

Result: **29/30 pairs pass**. The 1 failure is the
layer-0 non_spec capture that ALSO fails the single-batch
`test_sycl_matches_fla` with the identical max diff (8.31e-2) —
the documented pre-existing rung-3 capture non-determinism, not
caused by the mixed path.

## Rung 8 — reorder_input=False equivalence via layout permutation

All captures use `gqa_interleaved_layout=False`, so the
`reorder_input=False` kernel branch (used in production for
Qwen3-Next traffic) was never exercised by the existing replay
tests. Two helpers permute the captured projections into the
layout the kernel expects under `reorder_input=False`:

- `_qkvz_false_to_true(qkvz, cfg)` — repacks
  `[Q_block | K_block | V_block | Z_block]` (concatenated)
  into `[k_head_0:(q,k,v,z) | k_head_1:(q,k,v,z) | ...]`
  (interleaved). Mirrors the inverse of the `if constexpr
  (ReorderInput)` index math at `causal_conv1d.hpp:194-225`.
- `_ba_false_to_true(ba, cfg)` — repacks `[b | a]` into
  `[kh0_b | kh0_a | kh1_b | kh1_a | ...]`. Matches the
  `reorder_input=False` BA path at `causal_conv1d.hpp:131-143`.

`test_sycl_reorder_input_false_equivalence` runs the SYCL kernel
twice on the same capture: once with the captured layout and
`reorder_input=True`, once with the permuted layout and
`reorder_input=False`. Both calls use independent fresh pools so
the in-place state updates don't cross-contaminate. Outputs +
slot states must match within bf16 tol (atol=rtol=2e-2).

Result: **200/200 captures pass**. The
`reorder_input=False` code path is verified equivalent to the
`reorder_input=True` path on the same data.

## Notes

- Existing `is_mixed` xfail in `test_sycl_matches_fla` was changed
  to `pytest.skip` with a pointer to the new mixed-batch test,
  since `_mixed`-flavor captures don't exist in the dump anyway.
- The capture-name parser was tightened: flavor segments contain
  underscores (`spec_K4_min1_max1`, `non_spec`), so the prefix /
  flavor split now anchors on the 6-digit step token rather than
  splitting on the last underscore.
- No kernel rebuild needed; both tests run against the existing
  tick-40 `.so`.

## Source state

- `tests/kernels/xpu/test_spec_gdn_replay.py` extended with two
  new test functions, the unified-pool builder, the layout
  permutation helpers, and a robust capture-name parser.
- `.spec-gdn-progress.md` ladder: rungs 7 and 8 marked GREEN with
  a note on the synthesis approach. Status banner updated to
  "Rungs 3, 4, 5, 7, 8, 9, 11 GREEN".
- No source/kernel files modified.

## Open work after tick 46

Only rung 10 (e2e Qwen3.6+MTP-K3 with acceptance-rate
benchmarking) remains. That needs full vllm serve + reference
acceptance-rate harness, out of replay-harness scope.

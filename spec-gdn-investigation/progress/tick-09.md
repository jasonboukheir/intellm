2026-05-07 tick 9: bug is NOT IS_SPEC — it's a single-token native bug
that affects production silently.

**Disproof of "IS_SPEC bug"** (`vllm/.spec-gdn-narrow-bypass-spec.py`):
Ran the rung-4 capture two ways through the SAME native kernel:
- Run 1 (IS_SPEC=true, the test's normal call): 116 fails, max
  diff 2.64.
- Run 2 (IS_SPEC=false, num_decodes=1, qsl=[0,4],
  non_spec_state_indices=[load_slot]): **byte-identical to Run 1**
  (max delta between Run 1 and Run 2 = 0.0). Same 116 fails at the
  same v_dim positions [25, 33, 47, 53, 55, 60, 79, 83, 107, 124].
  IS_SPEC only affects writeback target; the chunk math is
  shared, and the bug is in the shared math.

**Bug at seq_len=1 too** (`vllm/.spec-gdn-narrow-seqlen1.py`):
Called the same capture as a regular single-token decode
(num_prefills=0, num_decodes=1, qsl=[0,1], no spec args):
- 30 / 4096 fails on `core_attn_out[token-0]`.
- Same v_dim set: `[25, 33, 47, 53, 55, 60, 79, 83, 107, 124]`.
- Worst: `(token=0, head=14, v=79): sycl=-0.209  fla=+1.016
  delta=1.225`.

**Implication**: the native `gated_delta_rule.hpp` chunk loop
silently mis-computes `core_attn_out` for **single-token decode**
in production. The bug has been there since the kernel landed.
Production hasn't noticed because (a) atol=2e-2 in the replay test
is much tighter than what end-to-end accuracy gates ever check,
(b) only ~0.7% of cells exceed it (mean diff ≈ 3.3e-3), and (c)
production never compares against FLA at this granularity. End-
to-end Qwen3.6 + MTP-K3 acceptance baselines (85.6/71.0/58.6) are
robust to this much per-cell drift.

This INVALIDATES rung 3's "code landed" claim — it was never
verified against FLA, and if it had been, it would have failed.
We should not have declared rung 3 done.

**Per-lane structural pattern** (decoded with sub_group_size=32,
group_size=256 → sg_per_group=8, v_dim_per_sg=4, v_dim_per_group=32,
num_v_bucket=4, j = v % 4):
- v=25 j=1, v=33 j=1, v=47 j=3, v=53 j=1, v=55 j=3, v=60 j=0,
  v=79 j=3, v=83 j=3, v=107 j=3, v=124 j=0.
- **j=2 NEVER fails. j=0 (2 cases), j=1 (3 cases), j=3 (5 cases).**
- Per-head distribution: same head can fail at one j and not at
  another in the same sg. So it's not a "broken sub-group", it's a
  per-(head, v) coordinate pattern with j=2 systematically
  protected.

**Hypothesis**: there's an off-by-one or unrolled-loop interleaving
quirk where state_local[j*kbs+i] for j≠2 picks up a wrong
contribution. The j=2 lane (inner state slice indices 8..11) is
in the "middle" of state_local[0..15] and may be the only
sub-slice the compiler doesn't reorder. Worth reading the
generated SPIR-V or experimenting with `#pragma nounroll` /
`volatile` annotations on `state_local`. The double-loop at
207-213 (decay+kv_mem) and 237-244 (state-update+res) interleaves
state_local writes and reads that depend on each other; if the
compiler hoists the write past the read for some j-lane, you'd
see exactly this pattern.

Worth also checking: line 156-158 in the no-init-state branch uses
`state_local[i * v_dim_per_sg + j]` (transposed index). Since
v_dim_per_sg == k_bucket_size == 4 for our build, the loop still
zeros every element, but the inconsistency could prime the
compiler to alias state_local with a different layout assumption
in the rest of the kernel — not impossible. Worth fixing the
inconsistency on principle even if it's not the root cause.

Next tick: read the disassembled SPIR-V (or LLVM IR) for the
spec-true template instantiation around the double-loops, focusing
on whether `state_local[j*4+i]` reads/writes use the expected
layout. If we can't get IR easily, try a minimal repro: a
self-contained SYCL test that replicates the kv-rule loop with
known inputs at v=79. Don't rebuild yet unless the experiment
absolutely requires it.


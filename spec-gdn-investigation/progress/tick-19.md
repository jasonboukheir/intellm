2026-05-07 tick 19: load-readback probe — bug is per-iter state_local
corruption, not load alone. Tick 18's "load" verdict was incomplete.

Wrote `vllm/.spec-gdn-load-readback.py`. Setup: `A_log = -100`,
`b = -100`. Sanity-checked on XPU (`vllm/.spec-gdn-g-beta-sanity.py`):
`g = 1.0` EXACTLY in fp32, `beta = 0.0` EXACTLY in fp32 (denormal
underflow flushed correctly; `1/(1+exp(100)) = 1/+inf = 0`). So
math says iter t = identity for all t.

Probe runs SYCL on the layer-0 worst capture, then compares state_post
across spec slots:

  iter pair      max abs delta   mean         hot (h, v)
  iter0 vs iter1  5.80           5.4e-4        839
  iter0 vs iter2  7.68           9.3e-4       1394
  iter0 vs iter3  9.22           1.1e-3       1760
  iter1 vs iter2  2.13           6.6e-4        687
  iter1 vs iter3  3.42           9.0e-4       1239
  iter2 vs iter3  1.87           5.5e-4        684

  iter 0 writeback vs pre-state (load defect): max 8.95, 9011 cells

The drift accumulates monotonically — each iter adds ~2 units of max
delta and ~350 hot (h, v) cells. With g=1, beta=0 exactly, NO
mathematical change is possible. Therefore the COMPILED kernel
corrupts `state_local` in the iter loop body even when arithmetic
is no-op.

**Refined model**:
- Load defect (iter 0 writeback vs pre-state, max 8.95) = state_local
  at end of iter 0. Includes iter-0's compute corruption on top of
  any load corruption.
- Per-iter drift (cross-iter, ~2 units per iter) = each iter's compute
  corrupts state_local further.
- These are very plausibly the SAME bug applied at each iter — a
  codegen issue that shifts state_local at j=3 (and other v-positions)
  every time we execute lines 210 (`state_local *= g`) or 241
  (`state_local += k_local * delta`).

**Concrete hypothesis**: memory aliasing between
`kv_mem[v_dim_per_sg]` and `state_local[v_dim_per_sg * k_bucket_size]`.
With v_dim_per_sg = 4 and k_bucket_size = 4, state_local has 16
elements indexed `j * 4 + i` for j∈[0,4), i∈[0,4). If the SYCL
compiler co-locates kv_mem and state_local (registers spilled to SLM
or otherwise overlapping), kv_mem[3] writes (line 211 `+=`) could
alias state_local[12..15] = the j=3 lane. v=79 sits at j=3, which
matches the bug's hot lane.

Top-10 v_dim by load-defect-incidence: v=79(31), v=33(12), v=1(11),
v=17(11), v=60(11), v=103(10), v=9(10), v=20(10), v=22(9), v=47(9).
The set is broader than tick-15's "magic numbers" because the readback
counts cells with |delta|>1e-3 (not >2e-2), but v=79 still dominates
(~3× the next).

Top-10 head by load-defect-incidence: h=14(116), h=19(115), h=18(112),
h=15(99), h=2(89), h=4(77), h=3(73), h=16(72), h=5(64), h=21(30).
Not localized — most heads see corruption, but heads 14, 15, 18, 19
are the worst (consistent with tick-14's h=14 ↔ h=15 pair finding,
extended to h=18/19).

**Tick 20 plan**: confirm or refute the kv_mem aliasing hypothesis.
Approach 1 (preferred): incremental rebuild with `volatile float
kv_mem[v_dim_per_sg]` (and `volatile float res[v_dim_per_sg]`) to
force the compiler to use distinct storage from `state_local`. Run
the load-readback probe. If iter0-vs-iter3 drops to bf16-noise level
(<1e-3), aliasing is confirmed; the fix is to declare these arrays
non-aliasable (volatile, or distinct type tagged with `[[clang::no_alias]]`,
or moved to local-memory with explicit allocation). Cost: incremental
ninja ~270-600s.

Approach 2 (also viable, slower): dump SPIR-V/LLVM IR for the
spec-true template instantiation; inspect addresses of state_local[12..15]
and kv_mem[3] in the generated code. Reveals the aliasing directly.
Cost: rebuild with `-save-temps` or equivalent, ~600-900s.

I'll pick Approach 1 for tick 20 — directly testable, low rebuild
cost, and a clean fix path if confirmed.

Tools added:
- `vllm/.spec-gdn-load-readback.py` (load+per-iter drift probe)
- `vllm/.spec-gdn-g-beta-sanity.py` (XPU sanity check on g=1, beta=0)


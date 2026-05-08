2026-05-07 tick 13: bug is at iter 0 — accumulation hypothesis rejected.

Wrote `vllm/.spec-gdn-iter-trace.py`. Trick: rung-4 has K=4 spec slots,
each holds the post-iter-t state (writeback at `spec_idx[batch, t]`),
so a single SYCL call gives all 4 iteration outputs without modifying
the kernel. Pyref runs the same chunk loop in fp32 and snapshots the
watched cell at every iter. Worst layer-0 v=79 cell auto-selected by
greatest |sycl-fla| across tokens × heads at v=79.

**Worst cell: layer 0, token=2, head=14, v=79.** Captured FLA = 0.488,
pyref (fp32) = 0.488 (byte-equal at this cell, confirming pyref ≡ FLA),
SYCL = -0.734. Delta 1.22 = the magic-number tick 12 saw across the
sweep.

Per-iter table (head=14, v=79, |state_post| over k):
  iter   g     beta    v_t   pyref|s|  sycl|s|  max-diff  pyref-core sycl-core
   0  0.998  0.936   8.125    13.73    12.87    **4.32**    0.903     -0.209
   1  0.998  0.954  -0.152    13.15    12.85    4.04        0.819     -0.248
   2  1.000  0.957  -0.007    13.50    12.86    4.36        0.488     -0.734
   3  0.972  0.907   0.000    13.31    12.46    4.06       -0.835     -0.898

**The bug enters at iter 0 with max abs diff 4.32 across the K=128
state lane. Iters 1-3 keep the same ~4 amplitude (because g≈1 and v_t
≈0 for the late tokens — the chunk loop is barely updating).**

This kills:
- The bf16-accumulation hypothesis (tick 12 idea): there's no
  accumulation happening. Iters 1-3 don't drift further.
- The "multi-token chunk loop bug" framing (tick 6, 8): it's
  fundamentally a per-iteration bug. Rung 3 captures pass purely
  because their iter-0 inputs don't trigger the defect, not because
  they iterate fewer times (they iterate the same 4 times — they
  just don't have the bad state magnitudes).

K-coord breakdown at the worst cell (iter t=2, head=14, v=79, slot 3):
many k positions show large divergences (sign flips at k=0, 115, 118,
122; near-zero collapses at k=92, 11; magnitude swaps at k=78, 102,
123). It's NOT one bad k-lane — it's many. State-wide miscompute at
this (head, v).

**Refined hypothesis**: SYCL's iter-0 compute at `(head=14, v=79)` is
mistakenly mixing data from another (head, v) coordinate. Possible
sources, in order of investigatability:
1. Initial-state load at `state_local[j*K + k] =
   ssm_state[load_slot * H*Hv*Hk + head*Hv*Hk + v*Hk + k]`
   (gated_delta_rule.hpp:144-149). If the index expression for the
   `j` lane gets the wrong v (e.g., v_bucket arithmetic off by one
   for some j), iter 0 starts from the wrong state. Could explain
   the 4.32 magnitude — v=78 or v=80's initial state is just as
   plausible a 13-magnitude state.
2. The reduce_over_group sum for `kv_mem[head, v]` (line 207-213),
   if some sub-group lane contributes a stale partial.
3. The rank-1 update writeback (`state_local[j*K+k] += delta * k_local[k]`),
   if the `delta` broadcast picks the wrong v lane.

Layer-0 v=79 dominates because layer 0's pre-state at v=79 has
magnitudes large enough (13+ in pyref) that a coordinate-mix bug
produces a 4-unit error. Other layers' v=79 magnitudes are smaller →
smaller error → fewer cells over tolerance.

Next tick (preferred): isolate WHICH stage of iter 0 introduces the
4-unit error.

Plan A (quick, no rebuild): synthesize a "pre-decay snapshot" by
calling SYCL with `state_init` at slot 3 manually pre-decayed (zero
out v=79's contribution and re-pack). Then SYCL's iter 0 starts from
known clean state. If SYCL still diverges from pyref, the bug is
post-load (decay/reduce/update). If SYCL agrees, the bug is in the
load.

Plan B (more direct, may need rebuild): add a SYCL-side debug dump
of `state_local[head=14, v=79, k=*]` immediately after the load and
again after the decay. Compare both against pyref's snapshots. This
needs `gated_delta_rule.hpp` instrumentation; preserve the
build-cache (vllm-xpu-rebuild). Time budget ~270-600s incremental.

Plan C (cheapest first): inspect what's at neighbor coordinates
(head=14, v∈{78, 80}) in pyref's initial state. If pyref's initial
state at (h=14, v=78) or (h=14, v=80) has values consistent with what
SYCL writes at (h=14, v=79), that's strong evidence for a v-index
off-by-one bug at the load.

I'll start with Plan C in tick 14 — fastest signal, no rebuild,
either confirms or rules out the cleanest hypothesis.


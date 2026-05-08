2026-05-07 tick 8: sentinel test rules out "kernel didn't run".

User asked the right gut-check question: are we sure the SYCL
kernel is actually running, or are most "passes" just bf16-noise
agreement with empty/uninitialized buffers? Sentinel test
(`vllm/.spec-gdn-narrow-sentinel.py`):
- Pre-fill `core_attn_out`, `z`, and the post-state SSM slots
  (1, 2, 3, 4) with `-1024.0` (bf16-exact). Run the spec kernel.
- Result: **0/16384 core_attn_out cells**, **0/16384 z cells**,
  **0/524288 ssm cells per spec slot** still hold the sentinel.
  The kernel writes every cell of every output it owns.

So the wrong values are real kernel computations, not
uninitialized memory passing through. Bug is in the math.

**Important re-read of progress**: tick 0/1's "Rung 3 (k=1 fully
accepted): kernel + dispatch + test landed" is a code-landing
note, not a validation note ("Status: Rung 3 code landed; awaiting
GPU replay verification."). Captures are all K=4 (no K=1 captures
exist), so rung 3 has **never been verified against an FLA
oracle**. The slot-1 (post-token-0) diff already has 3009 wrong
cells, so the bug manifests after a SINGLE per-token iteration —
which means it would also fail rung 3 if K=1 captures existed.

Implication: this isn't a "multi-token loop bug". It's a
**single-token bug in the IS_SPEC code path of the native
gated_delta_rule kernel** — the same path k=1 would have hit.
The fact that error grows per-token (3009 → 3732 → 3910 → 3933)
is a secondary amplification on top of the underlying single-step
defect.

Most likely defect surface: the `state_local` slice at j=3
(= last per-sg lane of v_dim) gets wrong values immediately on
the first iteration. Since the only thing that touches j=3
distinctly from j∈{0,1,2} is the lane-3 read in line 145
(loading initial state for v=head_v_dim_id+3), or a missed
bounds check, or sub-group lane-3 collision in
`reduce_over_group` for kv_mem/res.

Next tick (preferred): write a Python reference that mimics the
SYCL kernel's compute exactly (batch_id loop, sub_group simulation
via numpy/torch indexing) and run it on the captured inputs
(post-conv q/k/v). Diff against FLA's captured ssm_state_post.
If the Python ref passes, the SYCL kernel has a SIMD/sub-group
implementation bug. If it fails, the **algorithm itself** is
wrong — i.e., the IS_SPEC chunk loop in gated_delta_rule.hpp
diverges from FLA at the math level. This avoids the rebuild
entirely.

Heads-up: the q/k/v captures aren't in the payloads
(`projected_states_qkvz` is the pre-conv projection). To get
post-conv q/k/v we'd need to either capture them or replay just
the conv1d step. FLA's conv1d is at the same Python layer
producing those same intermediates; cleaner to compare gated
delta rule alone using FLA's own intermediates as ground truth
inputs. Plan: load `projected_states_qkvz`, run FLA's
`fused_recurrent_gated_delta_rule_fwd` (or its inputs) in
pure-Python to get reference q/k/v, then call SYCL gated_delta
rule with those. If still wrong, the bug is purely in
gated_delta_rule.hpp's IS_SPEC math.


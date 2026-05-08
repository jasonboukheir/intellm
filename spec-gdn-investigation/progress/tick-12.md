2026-05-07 tick 12: rung 3 GREEN, rung 4 value-sensitive, layer 0 dominant.

Wrote `vllm/.spec-gdn-bug-footprint.py` (uses the new
`_compute_fla_spec_oracle()` helper). Ran across all 170 spec captures
(90 rung-3 + 80 rung-4), aggregated per (rung, layer):

**Rung 3 (`spec_K4_min1_max1`, 1 token accepted)**: PASSES every layer.
Across all 30 layers × 3 captures = 90 captures, **fail_count = 0** in
every one. Max abs diff worst case 1.95e-3 (layer 0/37). This is a
material change from tick 8's claim that "the bug manifests after a
SINGLE per-token iteration": tick 8's evidence was vs the captured
oracle, which tick 10 proved unreliable. With the inline oracle, rung 3
is genuinely correct. **Rung 3 is now verified, not just "code landed".**

**Rung 4 (`spec_K4_min4_max4`, 4 tokens accepted)**: fails every layer
but with massive amplitude variance (per-layer captures × 3, except
layers 26/28/29/30/32/33/34/36/37/38 had 2):

  layer  0 :   237 fails, max 1.51 — **worst by far**, dominated by v=79 (159/237)
  layer  1 :    81 fails, max 0.37, top v=66
  layer  8 :   114 fails, max 0.21, top v=23
  layer 20 :    84 fails, max 0.17, top v=23
  layer 14 :    47 fails, max 0.05, top v=124,110,105
  layer 37 :    40 fails, max 0.46, top v=90,116
  layer 32 :     0 fails, max 1.66e-2 (under tolerance!)
  layer 22 :     1 fail,  max 2.10e-2
  layer 25 :     1 fail,  max 2.38e-2
  …

**Top-5 fail-count v_dim per layer differs by layer**: layer 0 is v=79
heavy; layer 1 is v=66 / 125 / 116; layer 8 is v=23 / 127 / 50 / 37;
layer 20 is v=23 / 127 / 37 / 27 / 16. So the tick-9 "magic v_dim set"
[25,33,47,53,55,60,79,83,107,124] was layer-0-specific; the union of
failing v_dim across all layers is 74 of 128 coords — i.e. **most of
the v dimension can fail in some capture**.

Implications:
- **The bug is value-sensitive, not coordinate-fixed.** Different inputs
  trigger fails at different v_dim positions. The j-lane structural
  hypothesis from tick 9 (j=2 protected) was an artefact of layer 0's
  specific inputs — it doesn't generalise.
- **Layer 0 has 30× the worst-cell amplitude of the median layer**. This
  is not a kernel-template bug (the template is layer-agnostic). It must
  be that layer 0's recurrence inputs (q, k, v, initial_state, g, beta)
  hit some numerical edge that other layers' inputs don't reach.
- The "rung-3 passes / rung-4 fails" delta is then about input
  *distribution* differences between captures (different prompts,
  different acceptance histories ⇒ different states) — NOT about the
  num_accepted_tokens value itself driving the kernel down a buggy
  path.

Tick 12 also checked the rung-3 case directly: `_compute_fla_spec_oracle`
correctly handles num_accepted=1 (loads `spec_idx[batch, 0]` as initial
state, runs 4 chunk iterations, writes 4 slots). All 90 captures passed,
so the chunk loop is fundamentally sound; the bug only manifests at
specific input-value combinations.

**Refined hypothesis for the codegen bug**: a numerical/precision issue
in the inner accumulator (`state_local *= g; kv_mem = sum(state_local
* k_local); state_local += (v - kv_mem) * beta * k_local`) under bf16,
where certain magnitudes of `state_local * g` cause the next
`state_local += ...` to land outside the bf16 representable range or
to lose significant bits during the reduce_over_group sum. **Layer 0's
v=79 inputs presumably push state_local at v=79 into a magnitude regime
where bf16 accumulation drops important bits**; later iterations
amplify the resulting drift.

This is testable without rebuilding: dump `state_local` snapshots from
the SYCL kernel at each iteration for a layer-0 v=79 cell and compare
against pyref's fp32 trace. If at iter 0 they match, iter 1 they
diverge, iter 2 they diverge more — then it's bf16 accumulation.
If they all match within bf16 noise — then it's a writeback aliasing
bug after all.

Updated ladder marker: **rung 3 = green** (verified, not just landed).

Next tick (preferred): instrument the layer-0 v=79 case for an
iteration-by-iteration accumulator trace.

Plan:
1. Pick the layer 0 capture with the worst layer-0 fail (any of the 3
   `layers_0_*_spec_K4_min4_max4` captures). Identify the worst single
   (token, head, v=79, *) cell in core_attn_out by re-running
   bug-footprint and printing per-cell.
2. Write `vllm/.spec-gdn-iter-trace.py`: pure-Python pyref_hpp from
   `cross-check.py` but instrumented to dump
   `(state_local[head, v, k], kv_mem[head, v], delta[head, v])` after
   each of the 4 chunk iterations, for the head and v=79 of interest.
3. Get the corresponding SYCL trace by also adding per-iter dumps to the
   pyref simulation — but for SYCL, easiest path: re-run with smaller
   chunk-loop length 1, 2, 3, 4 and observe the divergence at which
   iteration boundary.
4. Don't touch hpp / don't rebuild yet.

If the divergence appears at iter 0 (single iteration) — then there's
a per-iteration bug we missed and the chunk-loop hypothesis is wrong.
If it appears at iter ≥1 — then it's accumulation and we need to
inspect the bf16 storage of `state_local` between iterations
(line 145-168 / 207-285 in `gated_delta_rule.hpp`).

Alternative tick: profile rung 4 captures by `|max(state_local)|` over
the chunk loop and see if fail count correlates with state magnitude.
If yes → bf16 overflow is the culprit. Cheap test, narrow but
informative.


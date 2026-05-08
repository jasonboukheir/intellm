2026-05-07 tick 17: zero-state probe — bug splits load/decay vs downstream
(~94% / 6%).

Wrote `vllm/.spec-gdn-zerostate-probe.py` (Plan A from tick 13). Two runs
on the layer-0 worst capture (`000004_spec_K4_min4_max4`):

  Run             v=79 max   v=79 fail (atol=2e-2)   total core fail
  baseline (real init state) 1.223      52                       77/16384
  zerostate (ssm_pre=0)      **0.073**  27                       33/16384

Baseline reproduces tick 10/11/12's 1.22 max diff exactly. Zeroing the
SSM-state at the load slot collapses the v=79 max from 1.223 → 0.073
— a **17× drop**. With zero initial state, iter 0 reduces to
`state_post_update = (v_t * beta).unsqueeze(-1) * k_norm.unsqueeze(1)`;
load and decay paths are eliminated.

**Verdict**: ~94% of the v=79 fault lives in the load (lines 144-149)
or the decay (`state_local *= g`, lines 167-168). The remaining ~6%
residual at v=79 (max 0.073) is downstream — most plausibly the
`reduce_over_group` on `res` for `core_attn_out` at j=3 (since the
zerostate iter-0 store is per-element, not a reduction; the only
reduction left in the iter-0 path with zero init is `reduce_over_group`
over k for `res`).

Note the secondary v_dim positions: baseline has v=60, 33, 55, 47,
124 with max ~0.2-0.6; zerostate has v=124 (0.032), v=33 (0.031),
v=47 (0.019). The same set, but amplitudes drop 5-10× — the same
bug, scaled by initial state magnitudes at those (head, v) coords.

Two-component model:
- **Component 1 (~94%)**: SYCL mis-loads or mis-decays
  `state_local[j*K+i]` at the j=3 lane of certain v-positions, most
  prominently v=79. The defect is value-amplified: layer 0 has the
  largest |state_init[h, v=79, k]|, hence the loudest visible delta.
- **Component 2 (~6%)**: SYCL's `reduce_over_group(res, plus<>)` on
  `core_attn_out` may have a small per-(j,v) bias at j=3. Visible
  even when load/decay are eliminated. This is the same surface
  tick 9 hypothesised (j=3 lane bias) but the magnitude is much
  smaller than the headline bug.

This invalidates tick 16's "no logical bug, no j=3 special path"
conclusion as a closing-out — the source review correctly found no
explicit j=3 branch, but the codegen issue at j=3 in the load/decay
path is real and now numerically isolated.

Next tick (preferred): SUB-DIVIDE the ~94% load/decay component into
load-only vs decay-only.

Plan: synthesize an "A_log = -∞" capture variant where decay g ≡ 1.
- Set `A_log_test = torch.full_like(A_log, -100.0)` so
  `exp(A_log_test) ≈ 0`, hence `g = exp(0) = 1` for all heads.
- Re-run SYCL and FLA oracle with this modified A_log on baseline
  init state. Decay is neutralized; load is exercised.
- If v=79 still fails ~1.0+: bug is in the LOAD (lines 144-149).
- If v=79 collapses to ~0.0+: bug is in the DECAY (`state_local *= g`,
  lines 167-168). Decay is per-element so a "decay bug at j=3" would
  only fail if the compiler-unrolled multiply at j=3 picks up the
  wrong `g` broadcast.
- Cheap, no rebuild, single binary signal in <2 min.

Then tick 19: depending on tick 18's verdict, either dump the load
SPIR-V (if load is the culprit) or inspect the decay codegen
(if decay). Both require a rebuild with `-save-temps` — defer until
tick 18 narrows.

Tools added: `vllm/.spec-gdn-zerostate-probe.py` (reusable; modify
the input and call diff_summary / by_vdim helpers).


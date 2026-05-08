2026-05-07 tick 16: source review — no logical bug, no j=3 special path.

Read `csrc/xpu/gdn_attn/gated_delta_rule.hpp:1-300` and
`gated_delta_rule.hpp:340-520` (dispatch). Findings:

**Template instantiation** (confirmed for our build):
  `gated_delta_rule_kernel<T=bfloat16, StateT=float,
                          k_bucket_size=4, IS_SPEC=true>`

`k_bucket_size = head_k_dim / sub_group_size = 128 / 32 = 4`. With
`v_dim_per_sg = 4`, `state_local` is `float[16]` per work-item,
holding a 4×4 (v, k) tile. Each work-group has 8 sub-groups; each
sub-group covers 4 v-values × 128 k-values across its 32 lanes. Per
lane: 4 v-values × 4 k-values.

**v=79 mapping**: bucket 2 (`v_bucket_id=2`), sg_id=3, j=3,
head_v_dim_id=76, sg_local_id=19 contains k=76..79. 32 lanes covering
k=0..127 each touch j=3 of v=79. Confirms tick 4.

**The line-156 "transposed index" is NOT a bug.** With
k_bucket_size == v_dim_per_sg = 4, both `state_local[j*kbs+i]` and
`state_local[i*vps+j]` cover the same 16 flat indices. The
`!has_init_state` branch (line 152-159) zeros the array — order of
visit doesn't matter, all 16 elements are set to 0. And in our
scenario `has_init_state=true` (load slot is non-null), so the dead
branch doesn't execute anyway. Tick 9's "stylistic-but-not-bug"
description was correct; tick 15 over-promoted it.

**No j=3-specific code paths.** Every state_local access in the live
path uses `j * k_bucket_size + i`, with `j ∈ [0, v_dim_per_sg)` and
`i ∈ [0, k_bucket_size)`, both fully unrolled (`#pragma unroll`).
Live blocks reviewed:
- L142-149 load (j outer, i inner)
- L207-213 decay + kv_mem (j outer, i inner; interleaved write+read)
- L237-244 state-update + res (j outer, i inner; interleaved
  write+read)
- L274-281 IS_SPEC writeback (j outer, i inner)
- L293+ non-spec writeback (j outer, i inner)
- reduce_over_group calls on kv_mem and res are flat over
  v_dim_per_sg, no j=3 special case

**Within-lane k variation hint**: tick 13's k-coord breakdown for
(h=14, v=79, slot 3, iter t=2) showed lane 0 (sg_local_id=0,
k=0..3) had k=2 byte-equal between sycl/pyref (d=0.001) but k=0, 1,
3 each off by 0.27–4.11. If the bug were a lane-level constant
offset, all 4 of a lane's k's would be wrong. The mixed pattern
(one k correct, others wrong within a single lane) implies a
compiler-unroll-level oddity — e.g. one of the 4 unrolled `i`
iterations of the k loop has correct codegen and the others don't.
Source has all 4 iterations identical (only loop variable changes)
so this is purely a codegen issue.

**Implication**: source-level fix is unlikely. Tick 17 must use
instrumentation OR a source tweak that pokes the compiler's
codegen (volatile, barrier, alternate unroll). Both need rebuild.

Build cost reminder: full AOT rebuild ~1h45m (per
`feedback_xpu_build_cache.md`). For a single hpp edit, default to
`vllm-xpu-rebuild` (incremental ninja) which should land in
~270-600s if the cmake cache survives. Avoid `vllm-xpu-clean`.

Next tick (preferred): instrumented build with a debug side-buffer
that captures `state_local` immediately after the load (before
decay) for the layer-0 (h=14, head_v_dim_id=76) work-item. Compare
to pyref's `state_init[14, 79, 76..79]` (pre-decay). If they match
→ bug is downstream of load (in decay/kv_mem/update/reduce). If
they differ → bug is in the load itself.

Plan:
1. Edit `gated_delta_rule.hpp`: add an optional `float* dbg_buf` (and
   gate via a template flag or runtime nullptr) that, at the
   load-completion point, writes `state_local[0..15]` into
   `dbg_buf[lane * 16 + idx]` only for the targeted (batch,
   num_v_heads_id, head_v_dim_id) — guarded so non-debug runs
   pay nothing.
2. Wire dbg_buf through the dispatcher (`gated_delta_rule.hpp:357-385`)
   as an optional parameter; keep `nullptr` default.
3. Wire it through `gdn_attn_interface.cpp` and the SYCL custom op
   schema (or expose as a separate debug op to avoid touching the
   prod schema).
4. `vllm-xpu-rebuild` (incremental).
5. Run iter-trace with debug buffer attached; diff against pyref
   `state_init[14, 79, 76..79]` and `state_init[14, 76..79, 76..79]`.

Alternative cheap probe (no rebuild, won't fully resolve but adds
data): synthesize a capture variant where q=k=v=0 and beta=0 → after
iter 0, state_post = state_init * g (no rank-1 update). If the
post-state shows the v=79 anomaly even in this no-input case → bug
is in the state load + decay only. If anomaly disappears → bug is
in the kv_mem/update path. Caveat: q=k=0 hits an l2-norm divide-
by-zero (eps prevents NaN but values become tiny). Doable but
fiddly. Note for tick 17b only.

Tools used: read-only review of `gated_delta_rule.hpp:1-520` and
`gdn_attn_interface.cpp:170-310`.


2026-05-07 tick 6: dispatcher-revealed bug locus + ssm_state amplifies.

**Dispatcher truth** (`csrc/xpu/gdn_attn/gdn_attn_interface.cpp:242`):
```
if (!is_spec && num_prefills > 0) { chunk_gated_delta_rule_xe2(...); }
else { NATIVE_LAUNCHER; }   // calls gated_delta_rule.hpp
```
The earlier non_spec finding (4 ssm cells off at v=79/107) used
**chunk_xe2**, not the native kernel. The spec path hits the
**native kernel**, which has only ever been exercised at seq_len=1
(rung 3 k=1) — rung 4 k=3 is the first time its per-token loop
iterates >1 time. So the spec bug is in the native kernel's
multi-token loop / writeback, not in chunk_xe2.

**ssm_state_post for rung-4 capture (extends `narrow-rung4.py`):**
- Slots 1, 2, 3, 4 each have ~3000–3900 failing cells (~0.7%).
  max abs diff = **10.73** at `(head=19, v=79, k=113)` in slot 2:
  sycl=+0.049 vs fla=+10.78. Same cell in slot 4: sycl=+0.011 vs
  fla=+7.79. **SYCL writes near-zeros where FLA writes ~10**.
- Worst cells dominated by `v=79`, also `v=70/93/113/115/122/...`,
  spread across ~15 heads. Heads 14, 15, 16, 18, 19, 29 recur.
- Mean abs diff 2.7e-3 — the kernel is mostly right but has a wide
  sparse "leak" at specific (head, v, k) where state should be
  large.
- Pattern across slots: more cells fail in later slots (slot 1:
  3009, slot 2: 3732, slot 3: 3910, slot 4: 3933) → **error grows
  per-token**, consistent with state_local accumulating an error
  each iteration of the seq loop.

**Implication:** the native kernel's per-token state update (line
210 `state_local *= g`, line 241 `state_local += k_local * delta`)
or the per-token spec writeback (line 277-281) has a bug that
manifests only when the seq loop iterates multiple times. core_attn_out
divergence at v=79 is downstream of the wrong state_local.

Code review of `gated_delta_rule.hpp` (lines 87-300) so far found
nothing definitively wrong — index patterns symmetric between read
(145), spec write (277), non-spec write (296); kv_mem and res are
correctly zeroed inside the per-token loop; reduce_over_group calls
look correct; spec_stride wired correctly through interface.cpp:171
(= spec_idx.size(1) = 4 for our capture). One stylistic-but-not-bug
asymmetry in `state_local` zero loop (line 156-158, no-init branch
not taken in our scenario). Suspect either:
1. A missing per-token reset/sync that lets work-item-private state
   drift,
2. `reduce_over_group` being applied where each work-item still
   holds a per-k slice (so the broadcast carries a stale value),
   or
3. The IS_SPEC writeback at line 277 racing with the next iteration's
   read on the same memory (different slots, but same DRAM region
   if a subgroup masks out partials).

Next tick: read FLA's `IS_SPEC_DECODING` Triton kernel
(`vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py`,
search `IS_SPEC_DECODING`) and compare its multi-token state update
against gated_delta_rule.hpp's. Don't rebuild yet.


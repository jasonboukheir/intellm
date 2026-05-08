2026-05-07 tick 7: FLA reference vs SYCL native — compute IS identical.

Read FLA's `fused_recurrent_gated_delta_rule_fwd_kernel` in
`vllm/model_executor/layers/fla/ops/fused_recurrent.py:27-175`,
diffed against `gated_delta_rule.hpp:121-300`:
- Initial-state slot (FLA line 105-119): same as SYCL line 121-130
  — `ssm_state_indices[i_n, num_accepted_tokens-1]`. Layout
  `(slot, head, v, k)` matches kernel index pattern.
- Per-token loop (FLA line 122-165): structurally same as SYCL
  line 166-285 — decay (`b_h *= exp(b_g)` ↔ `state *= g`),
  kv_mem (`tl.sum(b_h * b_k, 1)`), delta (`b_v -= kv_mem; b_v *= beta`),
  state update (`b_h += b_v[:,None] * b_k[None,:]`), output
  (`b_o = tl.sum(b_h * b_q, 1)`), per-token store (line 152-165 vs
  SYCL line 261-285). Identical math, identical layout.
- Per-token writeback slot (FLA line 154-156): uses loop-i_t
  (`ssm_state_indices[i_n * stride + i_t]`). SYCL uses
  `(t - seq_start_offset)`. Same offset semantics.
- Reduction in SYCL (`reduce_over_group` on kv_mem and res) is
  cross-sub-group sum; matches FLA's per-thread total since each
  Triton thread holds the entire (BV, BK) block while SYCL slices
  it across 32 sub-group lanes.
- IS_SPEC vs !IS_SPEC in SYCL **only differs in writeback target**
  — chunk computation is identical. So a "k>1 chunk loop" bug
  would also affect non_spec multi-token decode IF that path were
  ever exercised. Production non_spec decode is single-token
  (k=1), so the bug went unnoticed until rung-4 spec.

Per-cell evidence for state load being correct (slot 4 initial):
the test's `_build_dense_pool` populates the dense pool from
captured `ssm_state_pre[remap[slot]]`, and FLA also read from
`ssm_state_pre[slot 4]` when capturing. So SYCL reads the same
floats FLA read. Initial state cannot be the divergence source.

**Refined hypothesis** (token-0 slot-1 already has 3009 wrong cells,
all centered at `v=79` across many heads): the inputs to
gated_delta_rule (q, k, v, b, a from native `causal_conv1d`) differ
from FLA's intermediates for tokens beyond the first — most likely
`v[t, head, v=79]` post-conv has a defect that only manifests when
the per-token loop iterates, since that's the only kv-rule input
that has a v_dim axis. `z` byte-equality proves the projection
copy is fine but proves nothing about the conv'd q/k/v.

Native `causal_conv1d` was only ever exercised with k=1 spec (rung
3) — same blind spot as gated_delta_rule's chunk loop. The
per-token compute uses `mixed_qkvz[(token_id - input_load_len + 1
+ i)]` (causal_conv1d.hpp:308-310) which advances correctly for
each token, but the conv-state load
(`conv_states_ptr[(Width - 1 - states_load_len + i) * conv_elems
+ reordered_elems_id + e]`, line 297-299) reads from the same
`states_id` for all tokens — ALL tokens read from the load slot
(spec_state_indices[batch_id, n_acc-1]), not from sequentially-
written per-token slots. That matches FLA's design (the conv
"state" is the prior-round end window, fixed for the whole spec
window). So conv1d itself shouldn't have a multi-token bug just
from the sequence-loading.

Next tick: actually dump the conv1d outputs (q, k, v, b, a) for
the rung-4 capture and diff vs an FLA replay of the same projections,
focusing on token 1, 2, 3 at (head=14, v=79). If v[1..3, 14, 79]
matches FLA, the bug is purely in gated_delta_rule's multi-token
loop; if it diverges, the bug is in conv1d. This requires
intercepting the q/k/v output of native causal_conv1d — easiest by
calling it directly in a small Python harness rather than
modifying the kernel. Don't rebuild yet — narrow first.


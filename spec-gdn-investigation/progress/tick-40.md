2026-05-08 tick 40: **THE FIX LANDS — rung 4 essentially GREEN**.

Implemented the FLA-aligned spec semantics in
`csrc/xpu/gdn_attn/causal_conv1d.hpp` in two steps:

**Step A — load slot + offset shift** (validated downstream
before changing writeback):
- Replaced `states_id = spec_state_indices[batch * spec_stride
  + (n_acc - 1)]` with `states_id = cache_indices[batch_id]`
  (FLA reads the slot at column 0 of indices, which the
  dispatcher passes as `non_spec_state_indices_tensor`).
- Added `load_offset_shift = n_acc - 1` (for spec) to the
  conv state row offsets in the partial-window load.
- After Step A: K=4 layer-0 core_attn_out max → 1.95e-3
  (from 1.22), ssm_state max → 0.030 (from 4.6-6). Conv
  state still wrong because per-token writeback unchanged.

**Step B — replace per-token writeback with rolled state
into slot[batch, 0]**:
1. Added `state_len` parameter to `causal_conv1d_kernel`
   (computed at the launcher as `conv_states_stride_0 /
   conv_elems`).
2. Resized `conv_states_tmp` to `(batch, state_len,
   conv_elems)` for spec (was `(batch, Width-1, conv_elems)`
   for non-spec).
3. Replaced the IS_SPEC writeback (lines 336-352) with two
   per-work-item writes into `conv_states_tmp`:
   - Always: `tmp[batch, state_len - K_call + t_in_seq, dim]
     = local_input[Width*e + Width-1]` (the latest input for
     this token).
   - For `t_in_seq < state_len - K_call`: also `tmp[batch,
     t_in_seq, dim] = pre[n_acc + t_in_seq]` (shift copy
     from the slot's pre-state).
4. Added `spec_state_roll_kernel<T>` (pass 2): copies
   `tmp[batch, state_len, dim]` → `slot[cache_indices[batch],
   state_len, dim]`. Race-free vs pass 1 because pass 1
   wrote only tmp; pass 2 is the sole writer of slot.
5. Plumbed `state_len` through the `kernel_launcher` and
   `KERNEL_LAUNCHER_IMPL` macro.

**Probe results (`vllm/.spec-gdn-coreattn-diff.py`)**:

Layer 0 K=4 (the original failure):
- core_attn_out max=1.95e-3, mean=1.0e-6 ✓ PASS
- conv_state per slot: **max=0.0** (byte-equal to FLA) ✓
- ssm_state per slot: max=0.030, mean=4e-6, only 1-2
  cells > 2e-2 atol (rtol gates them OK in allclose).

Layer 28 K=1:
- core_attn_out max=2.4e-4 ✓
- conv_state: **max=0.0** ✓
- ssm_state max=2.1e-3 ✓

**Full pytest ladder**: 398/400 pass (was 229/400 before fix).
- 169 spec captures fixed.
- 2 captures still fail with max diff 2.6e-2 (just barely
  over 2e-2 atol on cells where the expected value is
  near-zero, so rtol*|e|≈0). Likely 1-cell numerical noise
  in 2M-cell tensors. Investigate as part of rung 4 final
  cleanup or accept as bf16 precision boundary.

**Caller already correct**: `_gdn_xpu_spec_sycl_path` was
already passing `spec_state_indices_tensor[:, 0]` as the
`non_spec_state_indices_tensor` arg, so the kernel reads
the right slot. No caller changes needed.

**SSM kernel UNCHANGED**: per FLA's `fused_sigmoid_gating`
the SSM correctly uses per-token spec slots
(`spec_state_indices[batch, n_acc-1]` for load,
`spec_state_indices[batch, t]` for per-token writeback).
This was already correct in `gated_delta_rule.hpp`.

**Source files modified**:
- `csrc/xpu/gdn_attn/causal_conv1d.hpp` — spec load slot,
  load offset shift, rolled writeback, new
  `spec_state_roll_kernel`, plumbed state_len.

**Next ticks**:
- Tick 41: investigate the 2 remaining failures
  (`layers_0_..._000000_non_spec` and `..._000005_spec_K4_min4_max4`)
  — likely just bf16 precision; either tighten the
  comparison logic or document the boundary.
- After rung 4 fully green: continue ladder rungs 5-11
  (k=3 num_accepted=1, k=3 num_accepted=0, mixed batch,
  gqa_interleaved variants, determinism, e2e Qwen3.6 + MTP-K3
  acceptance, perf gate).

**Source state**: edited (Step A + Step B). Build state:
JIT-only with the fix. `_xpu_C.abi3.so` mtime 1778233695.

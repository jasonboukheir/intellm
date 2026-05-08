2026-05-08 tick 39: tick-38 fully verified; concrete fix
designed but NOT yet implemented (non-trivial restructure).

**Verification — capture conv_state_pre vs post**:
For `layers_0_..._spec_K4_min4_max4.pt`:
- slot 0 (= original slot 1): max diff pre↔post = 27.6,
  48416 cells changed.
- slots 1, 2, 3 (= original slots 2, 3, 4): pre == post,
  byte-equal, 0 cells changed.
- post[0] vs post[1..3]: max=33.0, mean=1.09 — slot 0 carries
  unique data; slots 1-3 are ignored.

**Confirms** tick 38's hypothesis: FLA writes only to
slot[batch, 0]; the other slots in the spec ring are
read-and-not-written by FLA. SYCL's per-token writeback to
slots [batch, 0..K-1] is mismatched.

**Caller audit** (`vllm/_xpu_ops.py:_gdn_xpu_spec_sycl_path`,
lines 521-600):
- Passes `non_spec_state_indices_tensor=spec_state_indices_tensor[:, 0].contiguous()`
  (column 0 of the spec ring).
- Passes `spec_state_indices_tensor=spec_state_indices_tensor`
  (full 2D tensor).
- The kernel currently uses the 2D tensor for IS_SPEC paths;
  the 1D version (`non_spec_*`) is set up but only used in
  `!IS_SPEC` paths.

So the data the SYCL kernel needs (the single slot at
column 0, plus state_len from the buffer stride) is already
available; the fix is internal to the kernel.

**Layout details (Width=4, K_max=4)**:
- Per slot in the conv buffer: `state_len = 6` time positions
  × `conv_elems = 8192` dim. Layout (slot, time, dim).
- `state_len = (Width-1) + (K_max - 1) = 6` (room for the
  width-of-conv history plus the spec drafts up to K_max).
- For a call with K spec tokens: post-state at slot[batch, 0]
  is `[pre[K..state_len-1], x[0], ..., x[K-1]]` (pre shifted
  left by K, drafts appended at end).

**The fix (target — `causal_conv1d.hpp`)**:

1. **State load (current lines 173-181)**:
   - Replace `states_id = spec_state_indices[batch * spec_stride + (n_acc-1)]`
     with `states_id = cache_indices[batch_id]` (this is the
     `non_spec_state_indices_tensor[batch]` = spec[batch, 0]).
   - Add time offset: when reading conv_states_ptr, add
     `(n_acc - 1) * conv_elems` to the row index — i.e.,
     load history at positions `(n_acc - 1) .. (n_acc - 1) +
     (Width - 2)` instead of `0 .. Width - 2`.

2. **State writeback (current lines 336-352)**:
   - Drop the per-token K-slot writeback.
   - Replace with a rolled-state writeback to
     `cache_indices[batch_id]`'s slot, covering all
     `state_len` time positions:
     - For row p in [0, state_len - 1]:
       - if p < state_len - n_tokens_in_seq:
         post[p] = pre[p + n_tokens_in_seq]   (shift-left)
       - else:
         post[p] = local_input[Width * e + Width - 1] for the
                   work-item where t_in_seq = p - (state_len -
                   n_tokens_in_seq).

3. **Race-condition consideration**:
   Each work-item is one (token, dim_chunk). The shift-left
   reads pre at row `p + n_tokens` and writes new[p]; if a
   later token's writeback writes new[p + n_tokens] before
   the read happens, the shift uses the wrong value.

   Two safe options:
   a. **Two-pass approach**: first kernel computes conv
      outputs (current logic) and writes per-token state to
      a *temp buffer* (`conv_states_tmp` already exists and
      has shape `(batch, width-1, conv_elems)`). Second small
      kernel reads pre-state from slot[batch, 0] and per-token
      values from the temp buffer, builds the rolled state,
      and writes back to slot[batch, 0]. Mirrors the
      non-spec prefill pattern that already uses
      `conv_states_tmp`.
   b. **Snapshot approach**: in the same kernel, work-item
      reads the OLD state position before any new writes can
      happen, stashes it in a register, then the writeback
      uses the stashed value. This requires a barrier between
      the snapshot and the new writes, which is not free
      across work-items.

   Option (a) is clearer and matches the kernel's existing
   non-spec pattern. Implement that.

**Caller-side implications**:
- `_gdn_xpu_spec_sycl_path` already passes spec[:, 0] as
  non_spec_state_indices. No change needed in the dispatcher.
- The `spec_state_indices_tensor` tensor is still needed: the
  SSM kernel uses it for the per-token writeback (which IS
  correct per FLA's SSM semantics). The conv kernel just
  switches to ignoring the 2D tensor in favor of the 1D one.

**Known unknowns / risks**:
- The two-pass design needs a barrier between pass 1 and
  pass 2. The current `causal_conv1d` C++ entry submits a
  single SYCL kernel; the rolling pass would need a second
  submission. Async ordering within the queue should suffice.
- `state_len` derived from `conv_states_stride_0 / conv_elems`
  must be passed/computed. With the existing kernel signature
  this is straightforward.
- Production FLA paths build the spec_state_indices tensor
  with column 0 always pointing at the "active" slot. Verify
  by reading `_gdn_xpu_python_path` and the runner's spec
  state-index allocation: if the convention differs (e.g.
  column 0 sometimes != active slot), the fix breaks.

**Implementation effort estimate**: ~2-3 hours for kernel
edit + caller verification + rebuild + test. Not appropriate
for a single tick; defer to tick 40.

**Tick 40 plan**:
1. Audit `_gdn_xpu_python_path` and runner spec_state_index
   allocation to confirm slot[batch, 0] = active slot.
2. Implement option (a) two-pass design in causal_conv1d.hpp.
3. Rebuild, run the K=1 layer-28 test (should pass), then
   K=4 layer-0 test (should also pass).
4. Run full rung 4 ladder (`pytest tests/kernels/xpu/test_spec_gdn_replay.py
   -k spec`).
5. If all green, mark rung 4 GREEN and continue to rung 5+.

**Source state**: pristine (no edits this tick). Build state:
pristine JIT-only, mtime 1778230290.

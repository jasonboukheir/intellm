2026-05-08 tick 38: **Architectural mismatch found** — SYCL
`causal_conv1d_update`'s spec semantics differ from FLA's. The
slot-per-token model SYCL uses doesn't match FLA's
rolling-history-within-1-slot model.

**Layout check** (capture `layers_0_..._spec_K4_min4_max4.pt`):
  - `conv_state_pre.shape = (4, 6, 8192)` — 4 slots × **6
    time positions** × 8192 conv dim.
  - 6 positions per slot is far more than Width-1=3 (the
    minimum needed for a single conv step).
  - `is_conv_state_dim_first = False` → kv_cache layout is
    `(slot, time, dim)` = (slot, 6, 8192).
  - `slot_indices = [1, 2, 3, 4]` — 4 spec slots referenced
    in this call.
  - `num_accepted_tokens = [4]` (full accept).

**FLA spec semantics** (causal_conv1d.py lines 838-925):
- Receives `conv_state_indices = spec_idx_dense[:, 0]` — only
  the first column → ONE slot per batch.
- Reads `prior_tokens = conv_states_base + (n_acc-1) *
  stride_tok` → reads at offset (n_acc-1) within slot's time
  axis.
- Width history positions loaded for the conv kernel.
- After conv, the new state slides the window by 1 position
  inside the same slot. Comment in FLA:
  > "Before forward: [history1, ..., historyM].
  >  After forward: [history2, ..., historyM, draft1, ...,
  >  draftN]. After acceptance with k tokens: [historyN-k+2,
  >  ..., historyM, draft1, ..., draftk]"

So FLA's model: **1 slot per sequence**, time axis ≥ M+N, the
single slot rolls forward each spec call.

**SYCL spec semantics** (causal_conv1d.hpp lines 173-184,
336-352):
- Receives `spec_state_indices` (full 2D tensor).
- Reads from `spec_state_indices[batch, n_acc-1]` →
  potentially a *different slot* per call depending on n_acc.
- Per-token writeback to `spec_state_indices[batch, t]` →
  EACH spec token writes to a *separate slot*.
- Each slot only holds Width-1 = 3 positions in the kernel's
  view (the writeback loop is `for (int i = 0; i < Width - 1;
  ++i)`).

So SYCL's model: **K slots per spec ring**, each holding only
Width-1 positions, separate slot per token.

**Why both fail in K=4 layer 0** (max conv 47, 39700 cells):
- SYCL reads slot 4's state (n_acc-1=3 → spec[batch, 3]=4).
  FLA reads slot 1's state (always [batch, 0]=1).
- The state at those slots was *prefilled* from the captured
  `conv_state_pre` which has slot-specific values. Slot 4 ≠
  slot 1, so SYCL and FLA work with different starting
  values.
- Output states at all 4 spec slots also diverge because
  SYCL writes K writebacks (one per token), while FLA writes
  back to slot 1 only (rolling within 1 slot).

**Why K=1 layer 28 fails on conv_state but not downstream**
(max 14.6 conv, ssm + core_attn_out PASS):
- SYCL reads slot 1's state (n_acc-1=0 → spec[batch, 0]=1).
- FLA reads slot 1's state (always spec[batch, 0]=1).
- **Same slot loaded** → same starting state → same q/k/v
  computation → same downstream ssm + core_attn_out (the
  passes confirm this).
- BUT: SYCL writes to slots 1, 2, 3, 4 (per-token); FLA writes
  to slot 1 only. The CONTENTS of slots 1-4 after the call
  differ — SYCL has per-token snapshots, FLA has rolling
  history in slot 1.

**This explains tick 36's "core_attn_out only fails for K=4"
finding cleanly**: when n_acc=1, both kernels start from the
same slot (slot 1), so downstream is identical — the
divergence is only in the writeback layout. When n_acc>1, SYCL
loads from a *different slot* than FLA, so downstream diverges
too.

**The conv_state divergence is a layout/semantic issue, NOT a
codegen / numerical-stability issue**. T17-T34's chase of IGC
codegen, RA pressure, fast-math, etc. was all on the wrong
axis.

**Tick 39 plan — design and implement the fix**:

The fix is to align SYCL's conv1d_update with FLA's semantics:
1. Read state from `spec_state_indices[batch, 0]` always (not
   `[batch, n_acc-1]`).
2. Within that slot's `state_len` time axis, use offset
   `(n_acc - 1)` to find the right starting position.
3. Load Width history positions from offset `(n_acc-1) +
   {0..Width-1}` → relative offsets 0..Width-1.
4. After conv, slide the window: write back to slot[batch, 0]
   at positions `[1..state_len]` shifted (drop oldest, add
   newest token's value).
5. **Drop the per-token slot writeback** — only the single
   slot[batch, 0] is updated.

Caveat — production code: this matches FLA exactly. The
caller's `_gdn_xpu_spec_*_path` may have been written against
the OLD per-token-slot model — need to verify it's not
expecting the per-token writeback. Check
`vllm/_xpu_ops.py:_gdn_xpu_spec_sycl_path` and the runner that
preps `spec_state_indices_tensor`.

**Caveat — the SSM path is different**: per
`fused_sigmoid_gating.py:105-110`, FLA's SSM does use
`spec_state_indices[idx_seq, n_acc-1]` for load, and writes
per-token to `spec_state_indices[idx_seq, i_t]`. So SSM's
per-token-slot model IS correct (matches T19-T35
observations). Only conv differs.

This asymmetry — SSM rolls per-token in K slots, conv rolls
within 1 slot — is a model-specific choice. The capture
layout supports both: ssm_state_pre is (4, 32, 128, 128) per
slot (no time axis), conv_state_pre is (4, 6, 8192) per slot
(time axis = 6).

**Source state**: pristine (no edits this tick).

2026-05-08 tick 36: rung-4 reorientation. The bug is real on
real data; sharpened the symptom but not the root cause yet.

**Probe**: `vllm/.spec-gdn-coreattn-diff.py` runs SYCL +
FLA-oracle on the layer-0 worst capture
(`tuple_..._layers_0_..._000004_spec_K4_min4_max4.pt`) with
`VLLM_XPU_USE_SYCL_SPEC_GDN=1` and diffs both `core_attn_out`
and per-spec-token `ssm_state_post`.

**core_attn_out** (shape T,H,V = 4,32,128):
  - max abs diff = 1.2227, mean = 1.76e-3
  - argmax (t=2, h=14, v=79): SYCL=-0.7344, FLA=+0.4883
  - **t=0,1,2 fail wildly; t=3 max only 0.0625 (essentially
    passes tol).** Per-token max:
    - t=0: 1.11 @ (h=14, v=79), 29 cells > 2e-2
    - t=1: 1.07 @ (h=14, v=79), 22 cells
    - t=2: 1.22 @ (h=14, v=79), 23 cells
    - t=3: 0.06 @ (h=14, v=79), 3 cells (PASS)
  - **v=79 dominates** (max 1.22; next v=60 max 0.60). Top-20
    cells are all at v=79 across multiple heads.
  - Top heads: h=14 (1.22), h=23 (0.97), h=15 (0.92), h=12
    (0.88). All show similar SYCL-≈-FLA-flipped-sign pattern.

**ssm_state_post per-spec-token** (spec slots 1-4, K=4):
  - **All 4 tokens diverge** (NOT just t=0-2):
    - t=0 slot=1: max=4.63, mean=1.17e-3, 2728 cells
    - t=1 slot=2: max=4.04, mean=1.38e-3, 2909 cells
    - t=2 slot=3: max=6.00, mean=1.64e-3, 3128 cells
    - t=3 slot=4: max=4.98, mean=1.38e-3, 2834 cells
  - State writeback is wrong at *every* iteration. ~0.5% of
    cells (per slot, of 32×128×128) blow tolerance.

**Apparent contradiction (t=3 mystery)**:
- ssm_state at t=3 diverges by max 4.98 (4.7% of 32×128×128
  cells over tolerance).
- But res at t=3 matches FLA (max 0.06, only 3 cells).
- Mathematically, `res = sum_k(state * q)`. If state diverges
  by 4-5 at some (v, k), but q at those k is ≈ 0, the products
  cancel and res lands near FLA. So the divergence locations
  in state are *exactly* where q is small at that token —
  which is plausible but not yet verified.

**What this rules out (so far)**:
- Not the act_sigmoid overflow bug (T34): that needed b=-100.
  Real b values are normal. Sigmoid is fine.
- Not "kernel didn't run" (T8 sentinel): kernel runs and
  produces non-trivial output.
- Not "load IS_SPEC slot wrong" (lines 121-131 logic looks
  right; load_idx = n_acc - 1 = 3 with K=4 fully accepted,
  matches FLA's `i_t = n_acc - 1`).

**What's still suspect**:
- **Per-token state writeback** at lines 265-285. ssm_state
  diverges at every slot, so writeback may be storing wrong
  values OR storing correct values at the wrong layout.
- **q_local / k_local reduction** (lines 189-198, qk l2norm
  + scale). Bug expressing on specific v_dim could mean a
  per-head reduction issue.
- **Reduction across subgroup** (lines 216, 248). If the
  `reduce_over_group` implementation has subgroup-size
  sensitivity not yet considered.

**Tick 37 plan — distinguish writeback bug vs computation
bug**:
1. **State writeback isolation probe**: dump `state_local` at
   the END of the t-loop body (just before writeback), via an
   atomic write to a debug buffer keyed by (batch, head, v, k,
   t). Compare against FLA's b_h at the same (batch, head, v,
   k, t).
   - If SYCL state matches FLA at write-point → writeback
     itself is buggy (line 277-281 indexing or a race).
   - If SYCL state already diverges at write-point → bug is
     in the *computation* (lines 200-244), and we need to
     bisect that.
2. As a cheaper alternative to a debug buffer, do a probe with
   K=1 spec (no accumulation). If SYCL agrees with FLA on
   single-token, the bug is in the t-loop iteration path.

3. **q/k l2norm probe**: temporarily disable
   `use_qk_l2norm_in_kernel` in source (replace q*=rsqrt(qsum)
   with no-op). If state diff drops, the qk-l2norm path is
   suspect.

**Tools written**:
- `vllm/.spec-gdn-coreattn-diff.py` — the diff probe used in
  this tick. Reusable for any spec capture.

**Source state**: pristine. Build state: pristine JIT-only,
mtime 1778230290.

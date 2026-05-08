2026-05-08 tick 44: rung 6 (n_acc=0) synthetic probe —
**defensive code verified, but FLA itself has UB at n_acc=0**.

**Probe**: `vllm/.spec-gdn-nacc0-probe.py` takes a min1_max1
capture (`layers_0_..._000001_spec_K4_min1_max1.pt`), forces
`num_accepted_tokens = [0]`, runs both SYCL and FLA inline
oracle, compares.

**Result**:
- core_attn_out: max=0.25, mean=1.7e-4 (over tol).
- conv_state slot 1: max=31.6 (diverged).
- conv_state slots 2-4: max=0.0 (byte-equal — only slot 1 is
  written by both).
- ssm_state all slots: max=4.6-5.6 (diverged).

**Why this isn't a real failure**:
- FLA's `fused_sigmoid_gating.py:106` computes
  `i_t = tl.load(num_accepted_tokens + i_n) - 1`. With n_acc=0,
  `i_t = -1`.
- Then `state_idx = tl.load(ssm_state_indices + i_n *
  stride_indices_seq + (-1))`. For i_n=0, this reads
  `ssm_state_indices + (-1)` — one element BEFORE the
  indices array. **UB in C/Triton.**
- The same UB exists in FLA's conv1d_update (line 853):
  `conv_state_token_offset = num_accepted - 1 = -1`. The
  subsequent `prior_tokens = base + (-1) * stride_tok` reads
  one position before the slot's time axis — UB.

So FLA and SYCL diverge at n_acc=0 because BOTH have
implementation-defined behavior in this corner case, not
because either is "wrong".

**SYCL's defensive choice (already in place)**:
- conv1d (causal_conv1d.hpp:165-172): `spec_skip_init_load
  = true`; `local_input` zero-init for the load region;
  shift-copy in the rolled writeback uses `n_acc_safe = 0`,
  so it reads pre-state at positions [0, state_len-K_call-1]
  (= [0, 1] for state_len=6, K_call=4).
- ssm (gated_delta_rule.hpp:122-125): `load_slot=0`,
  `has_init_state=false`. State_local zero-init.

**Production reality check**: in actual Qwen3.6 + MTP-K3
traffic, the runtime guarantees `num_accepted_tokens >= 1` for
spec-decoding calls (the previous round always accepts the
"bonus" token at minimum). The n_acc=0 path is reachable only
on synthetic input or a fault scenario, and the defensive
behavior (start from zero state) is the safest possible.

**Decision**: mark rung 6 as **DEFENSIVE PASS** — the
behavior is safe (deterministic, no crash, no use of
uninitialized memory), but cannot be byte-matched to FLA
because FLA itself is undefined at n_acc=0. Note this in the
ladder.

**Tick 45 plan**: assess rung 7 (mixed batch). The captures
in `/tmp/spec_gdn_captures` are all single-flavor (non_spec
or spec only). Two paths to make rung 7 testable:
A. Capture mixed-flavor traffic during a real Qwen3.6 +
   MTP-K3 run (re-run with the dump hook).
B. Synthesize a mixed batch by *concatenating* a non_spec
   capture and a spec capture in the harness, replicating
   the Python-side split that
   `_gdn_xpu_spec_sycl_path:605-` does in production.

Option B is cheaper and exercises the same kernel paths that
production uses (each call still goes through one IS_SPEC=true
kernel + one IS_SPEC=false kernel via the dispatcher's
`if num_prefills + num_decodes > 0` branch).

If neither A nor B fits a single tick, defer rung 7 and
move to rung 11 (perf gate) — that's a benchmark-style
measurement on existing captures that doesn't need new
captures, just timing infrastructure.

**Tools added this tick**:
- `vllm/.spec-gdn-nacc0-probe.py` — synthetic n_acc=0
  override probe; reusable for any min1_max1 capture.

**Source state**: tick-40 edits in place. No new edits.

2026-05-08 tick 43: **Rung 9 GREEN — SYCL spec path is
byte-deterministic.**

**Probe**: `vllm/.spec-gdn-determinism.py` runs the SYCL spec
gdn_attention path on a single capture 10 times, with a fresh
`_build_dense_pool` per run, and diffs `core_attn_out`,
`conv_state_post`, `ssm_state_post` against the first run.

**Result on layer 0 K=4 min4_max4** (the worst case from rung
4):
  - All 10 runs: **core=0.0000e+00, conv=0.0000e+00,
    ssm=0.0000e+00.** Byte-equal across the board.

**Result on layer 28 K=1 min1_max1** (the rung-5
representative):
  - Same: byte-equal on every iteration.

**Implication**: the SYCL kernel produces deterministic output
under a fixed input. No autotune cache, no reduction-order
non-determinism, no race effects — the new
`spec_state_roll_kernel` (tick 40) is also race-free as
designed.

**Tool added**: `vllm/.spec-gdn-determinism.py` — re-usable
for any capture; honors `SPEC_GDN_CAPTURE` env override and
`SPEC_GDN_DETERMINISM_RUNS` (default 10).

**Status — ladder progress after tick 43**:
- Rung 1, 2, 3, 4, 5, 9: GREEN.
- Rung 6 (n_acc=0): defensive code in place; no captures yet.
- Rung 7 (mixed batch): no captures; harness needs Python-side
  split for mixed.
- Rung 8 (gqa_interleaved=True): no captures with this config.
- Rung 10 (e2e Qwen3.6+MTP-K3): defer to dedicated tick.
- Rung 11 (perf gate): defer.

**Next tick (44) plan**: assess rung 6 — write a synthetic
n_acc=0 probe that takes a min1_max1 capture and forces
`num_accepted_tokens=0`, then runs SYCL and compares against
either FLA or the captured oracle. The kernel's defensive
`spec_skip_init_load` path is exercised here. If green, mark
rung 6 GREEN with a note that empirical verification used a
synthetic override rather than a real capture.

**Source state**: tick-40 edits in place. No new edits this
tick.

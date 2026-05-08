2026-05-08 tick 42: Rung 5 GREEN; assess rungs 6-11 testability
under current captures.

**Filtered pytest verification of tick-40 fix**:
- `-k spec_K4_min1_max1`: **180 passed, 220 deselected** (= 90
  captures × 2 tests = 180; all pass).
- `-k spec_K4_min4_max4`: **159 passed, 1 failed, 240 deselected**
  (= 80 captures × 2 = 160; 1 failure is the layer-0-000005
  bf16-noise outlier from tick 41).

→ Rung 5 (k=3 num_accepted=1) is fully GREEN. The tick-40 fix
covers both rung 4 (n_acc=4) and rung 5 (n_acc=1) without any
case-specific code paths — the n_acc parameter just controls
the `load_offset_shift` and the rolled-state shift positions
naturally.

**Rungs 6-11 — testability assessment under current captures**:

- Rung 6 (k=3 num_accepted=0): **no captures available**. The
  defensive code is in `causal_conv1d.hpp`
  (`spec_skip_init_load = true` when n_acc <= 0;
  `local_input` zero-init when not loading; the rolled
  writeback shift uses `n_acc_safe = max(n_acc, 0)`). Logic
  is in place and exercises a clean code path. Verifying
  empirically requires re-capturing with traffic that triggers
  num_accepted=0 (e.g. by forcing all spec tokens to be
  rejected in a synthetic test). Defer.

- Rung 7 (mixed batch): captures don't exist in the current
  `/tmp/spec_gdn_captures` (all are single-flavor). The test
  harness `xfails` the `_mixed` case at lines 489-495 with
  "Mixed-batch spec captures need the Python-side split that
  lives in `_gdn_xpu_spec_sycl_path`". Real fix: capture
  mixed-flavor traffic during a Qwen3.6 + MTP-K3 run that
  includes some prefill and some spec sequences in the same
  step.

- Rung 8 (gqa_interleaved_layout ∈ {True, False} ⊗ above):
  current captures have `cfg["gqa_interleaved_layout"] =
  False` (so kernel runs with `reorder_input = True`). To
  test `True`, need a model with that config, OR a synthetic
  test that flips the config on existing captures. Defer.

- Rung 9 (determinism): can be tested under current captures
  by running a spec capture 10× and confirming byte-equal
  output + state. Cheap probe; do this in a follow-up tick
  before moving to e2e.

- Rung 10 (e2e Qwen3.6 + MTP-K3): full inference run. Needs
  the model loaded; runtime infrastructure separate from this
  test ladder. Defer to dedicated rung-10 tick.

- Rung 11 (perf gate): SYCL spec tok/s ≥ FLA spec tok/s.
  Benchmark run. Defer.

**Recommendation for tick 43**: Run determinism probe (rung 9)
on a spec capture — 10× call same SYCL kernel, compare byte
outputs. If green, mark rung 9 GREEN. Then assess captures
needed for rungs 6/7/8, write a synthetic-capture generator
or document the gap.

**Source state**: tick-40 edits in place. No new edits this
tick. Build state: pristine JIT-only, mtime 1778233695.

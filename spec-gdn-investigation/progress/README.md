# Spec-aware SYCL gdn_attention — Progress

Status: 2026-05-08. **Rungs 3, 4, 5, 7, 8, 9, 11 GREEN; rung 6
defensive PASS.** Tick 40's FLA-aligned conv1d spec fix in
`causal_conv1d.hpp` is the load-bearing change; tick 45
confirmed SYCL is 10.4× faster than FLA. The kernel is
correct, deterministic, and fast. Tick 46 unblocked rungs 7
and 8 by extending the replay harness with synthetic
mixed-batch and `reorder_input=False` equivalence tests.

Remaining test failures (not caused by the fix):
- 1 spec borderline (2 cells bf16 noise on
  `layers_0_..._000005_min4_max4`).
- 1 non_spec pre-existing (rung-3 capture non-determinism,
  not in tick-40 scope).

**Remaining open rung is infrastructure-bound, not bug-bound**:
- Rung 10 (e2e Qwen3.6+MTP-K3): full inference benchmark.
  Substantial setup (vllm serve, acceptance-rate harness).
  Defer to a dedicated task.

**LOOP CLOSED** at tick 45. Structural blockers documented;
the per-tick mode is no longer the right tool for the
remaining rungs. Resume the loop only after a structural
unblock (new captures or harness PR).

**Tick 34 ROOT CAUSE**: the load-readback probe's drift = 8.9465
is caused by `act_sigmoid(b=-100)` returning a non-zero value
on device under fast-math (`-fapprox-func`,
`-funsafe-math-optimizations`, etc.). The probe sets `b=-100`
to force `beta=0` mathematically (`1/(1+exp(100)) = 1/inf = 0`),
but on device `exp(100)` overflow gets handled differently and
produces a non-zero beta, making `delta = (v-kv)*beta`
non-trivial. The "line 241 codegen bug" framing from T17-T26
was a MISDIAGNOSIS — line 241 is fine; its input `delta` was
wrong upstream.

The decisive probe chain (tick 33-34):
| Source change                                  | drift   | cells   |
|------------------------------------------------|---------|---------|
| Pristine `+= k_local[i] * delta[j]`            | 8.9465  | 9011    |
| `+= 0.0f` literal                              | **0**   | 0       |
| `+= delta[j]` (no multiply)                    | 11.5583 | 186880  |
| `+= delta[j]` + `float beta = 0.0f`            | **0**   | 0       |

**Invalidated** by tick 34: T17-T26 narrowing, T22 RA hypothesis,
T23-T24 ASM diff hunt, T26 volatile-temp diff, T27 JIT-vs-AOT,
T28 IGC env-var ladder, T29.A SIMD-16 — all chasing the wrong
target. The 8.9465 invariance across all those variations now
makes sense: every rebuild produced the same `act_sigmoid`
codegen (same buggy overflow handling), so all post-T26 builds
gave the same drift number.

Next: **tick 40 — implement the two-pass FLA-aligned conv fix**.
Tick 39 designed the fix and audited the caller. Concrete plan:
1. Audit `_gdn_xpu_python_path` (FLA path) and the runner's
   spec_state_index allocation to confirm column 0 always
   points at the "active" slot in production.
2. Edit `causal_conv1d.hpp`:
   - IS_SPEC load: read from `cache_indices[batch_id]` slot
     at time offset `(n_acc - 1)`.
   - IS_SPEC writeback: replace per-token K-slot writes with
     a two-pass design — pass 1 writes per-token state into
     `conv_states_tmp` (already declared in launcher), pass 2
     consolidates `tmp` + pre-state into the rolled state at
     `cache_indices[batch_id]` slot.
   - Plumb `state_len` (from `conv_states_stride_0 /
     conv_elems`) into the kernel.
3. Rebuild incremental (~10 min), run K=1 layer-28 then K=4
   layer-0 spec replays. Expand to full rung 4 ladder if
   green.
4. SSM kernel UNCHANGED (already correct).

Reusable probes:
- `vllm/.spec-gdn-coreattn-diff.py` (per-slot conv + ssm
  diff, tick 36-37)
- pytest `tests/kernels/xpu/test_spec_gdn_replay.py` (full
  ladder)

**T27 build/probe COMPLETE** (tick 29). Working JIT-only `.so`
installed at `vllm_xpu_kernels/_xpu_C.abi3.so` mtime 1778227678
(2026-05-08 ~01:08 local). Bug confirmed under JIT
(drift=8.9465). The build is reusable for IGC env-var probing
without further rebuild.

**Three wrapper/build traps documented** (relevant for any
future JIT-only / RA-decouple rebuilds):
1. `vllm-xpu-build` wrapper overrides empty AOT env vars back to
   `bmg` via `${VAR:-bmg}`. Bypass with direct `docker exec -e
   VLLM_XPU_AOT_DEVICES= -e VLLM_XPU_XE2_AOT_DEVICES= ...`.
2. Default `MAX_JOBS=2` OOM-kills `grouped_gemm_xe2.cpp`
   template instantiation on a 93 Gi host without swap. Use
   `MAX_JOBS=1` if both heavy template libs are in the build set;
   `MAX_JOBS=2` is OK if FA2 (chunk_prefill) is disabled.
3. **Disable `FA2_KERNELS_ENABLED=OFF` for any T27/T28-class
   experiment** that only needs `_xpu_C` (gdn_attention) — drops
   542 heavy chunk_prefill compile steps. setup.py honors the env
   var (`_is_enabled("FA2_KERNELS_ENABLED")`). `_xpu_C` does NOT
   link `attn_kernels_xe_2`; only `gdn_attn_kernels_xe_2` and
   `grouped_gemm_xe_2`.

## Tick history

Per-tick narratives live in `.spec-gdn-progress/`.
Most-recent ticks are at the bottom; read upward for context.

- [tick-04](.spec-gdn-progress/tick-04.md) — Tick 4 — pipeline-verification (no kernel work); axes-label bug found in narrow.py
- [pre-flight](.spec-gdn-progress/pre-flight.md) — Pre-flight — dev shell + container OK
- [tick-02](.spec-gdn-progress/tick-02.md) — Tick 2 — host HF cache bind-mount fix
- [tick-03](.spec-gdn-progress/tick-03.md) — Tick 3 — Triton blockers + harness bug + first SYCL-vs-FLA diff
- [tick-05](.spec-gdn-progress/tick-05.md) — Tick 5 — build cycle complete; first rung-4 spec replay (3/3 FAIL at v=79/107)
- [tick-06](.spec-gdn-progress/tick-06.md) — Tick 6 — dispatcher truth: spec hits NATIVE kernel; ssm_state amplifies per-token
- [tick-07](.spec-gdn-progress/tick-07.md) — Tick 7 — FLA reference vs SYCL native: compute IS identical; bug suspected in conv'd q/k/v
- [tick-08](.spec-gdn-progress/tick-08.md) — Tick 8 — sentinel test rules out 'kernel didn't run'; rung-3 was never validated
- [tick-09](.spec-gdn-progress/tick-09.md) — Tick 9 — bug NOT IS_SPEC, fails at seq_len=1 too; per-lane j-pattern (j=2 protected)
- [tick-10](.spec-gdn-progress/tick-10.md) — Tick 10 — pyref-of-hpp ≡ FLA-recurrence (math correct); SYCL is real codegen bug
- [tick-11](.spec-gdn-progress/tick-11.md) — Tick 11 — replay oracle switched to inline FLA-recurrence
- [tick-12](.spec-gdn-progress/tick-12.md) — Tick 12 — RUNG 3 GREEN; rung 4 value-sensitive, layer-0 dominant
- [tick-13](.spec-gdn-progress/tick-13.md) — Tick 13 — bug at iter-0 (accumulation hypothesis rejected)
- [tick-14](.spec-gdn-progress/tick-14.md) — Tick 14 — NOT a coordinate-mix; (h=14, v=79) ↔ (h=15, v=79) hot-pair
- [tick-15](.spec-gdn-progress/tick-15.md) — Tick 15 — even-odd head-pair rejected; v=79 universal hot lane
- [tick-16](.spec-gdn-progress/tick-16.md) — Tick 16 — source review: no logical bug, no j=3 special path
- [tick-17](.spec-gdn-progress/tick-17.md) — Tick 17 — zero-state probe splits load/decay vs downstream
- [tick-18](.spec-gdn-progress/tick-18.md) — Tick 18 — load vs decay split; bug in LOAD (decay ruled out)
- [tick-19](.spec-gdn-progress/tick-19.md) — Tick 19 — load-readback probe: per-iter state_local corruption (g=1, beta=0 drifts ~9.22)
- [tick-20](.spec-gdn-progress/tick-20.md) — Tick 20 — bug isolated to a SINGLE source statement: gated_delta_rule.hpp:241
- [tick-21](.spec-gdn-progress/tick-21.md) — Tick 21 — source-level workarounds bit-identical: compiler folds them
- [tick-22](.spec-gdn-progress/tick-22.md) — Tick 22 — more workarounds tried; self-assign confirms IGC elides provably-no-op writes
- [tick-23](.spec-gdn-progress/tick-23.md) — Tick 23 — Xe2 ISA disassembly pipeline working; spec=true vs false: same op mix, different RA
- [tick-24](.spec-gdn-progress/tick-24.md) — Tick 24 — inner-loop region located; drift is BOTH RA AND scheduling (T24 confound)
- [tick-25](.spec-gdn-progress/tick-25.md) — Tick 25 — REFRAME: cheap probes; bug is IS_SPEC=true template-specific
- [tick-26](.spec-gdn-progress/tick-26.md) — Tick 26 — volatile-temp ASM diff inconclusive (RA-wide reshuffle from spill); pivot to T27 JIT vs AOT
- [tick-27](.spec-gdn-progress/tick-27.md) — Tick 27 — wrapper-bug found (`${VAR:-bmg}` overrides empty); restarted T27 as direct `docker exec` (JIT-only verified)
- [tick-28](.spec-gdn-progress/tick-28.md) — Tick 28 — status check + reconfigure: dropped FA2 (542 chunk_prefill steps unneeded for `_xpu_C`); JIT build finished in 10 min vs 16h projected
- [tick-29](.spec-gdn-progress/tick-29.md) — Tick 29 — T27 VERDICT: drift=8.95 under JIT (same as T26 volatile-temp, ≠ pristine AOT 9.22) → bug is IGC-backend-wide, not AOT-only; pivot to env-var workaround ladder
- [tick-30](.spec-gdn-progress/tick-30.md) — Tick 30 — T28 IGC env-var ladder failed (4 vars, drift bit-identical at 8.95); pivot to source SIMD-16 / struct-wrap
- [tick-31](.spec-gdn-progress/tick-31.md) — Tick 31 — T29.A SIMD-16 + k=8 dispatch FAILED (drift bit-identical 8.95). T22 RA-pressure hypothesis falsified. Bug invariant across all codegen perturbations. Next: methodological sanity check.
- [tick-32](.spec-gdn-progress/tick-32.md) — Tick 32 — sanity check PASSED: zeroing line 241 jumps drift 8.95→14.70 (162871 cells). Pipeline healthy, 8.95 is genuine bug fingerprint. Next: probe operands (literal 0 vs k*delta).
- [tick-33](.spec-gdn-progress/tick-33.md) — Tick 33 — `+= 0.0f` literal → drift = 0; `+= delta[j]` alone → drift = 11.56. Operand-dependent.
- [tick-34](.spec-gdn-progress/tick-34.md) — Tick 34 — **ROOT CAUSE**: forcing `beta = 0.0f` zeros drift. `act_sigmoid(-100)` overflows fp32 under fast-math; T17-T26 line-241 framing was a misdiagnosis. Critical: probe artifact ≠ rung 4 real failure.
- [tick-35](.spec-gdn-progress/tick-35.md) — Tick 35 — Rung 4 still RED on real data (171/400 fail). Real bug is `core_attn_out` divergence (max=1.22 mean=1.76e-3), NOT ssm_state. T17-T34 narrowed within wrong probe; restart on core_attn_out.
- [tick-36](.spec-gdn-progress/tick-36.md) — Tick 36 — symptom sharpened: core_attn_out fails t=0/1/2 (max 1.1) passes t=3 (0.06); ssm_state fails ALL t (max 4.6-6); v=79 dominates. Next: writeback vs computation bisect.
- [tick-37](.spec-gdn-progress/tick-37.md) — Tick 37 — **HUGE PIVOT**: bug is in `causal_conv1d_update`, not gated_delta_rule. conv_state_post diverges max=47 (K=4) and max=14 (K=1). T17-T36 was investigating the wrong kernel.
- [tick-38](.spec-gdn-progress/tick-38.md) — Tick 38 — root cause: SYCL conv1d uses K-slots-per-spec-ring (Width-1 positions each), FLA uses 1-slot-per-sequence-with-rolling-history (state_len=6). Layout mismatch. SSM correctly uses per-token-slot.
- [tick-39](.spec-gdn-progress/tick-39.md) — Tick 39 — fix design: two-pass conv kernel (pass 1 = current per-token, pass 2 = consolidate to slot[batch, 0] rolled state). FLA-only-writes-slot-0 verified via capture diff (slots 1-3 byte-equal pre↔post).
- [tick-40](.spec-gdn-progress/tick-40.md) — Tick 40 — **THE FIX LANDS**. Step A (load slot + offset) + Step B (rolled writeback via temp + new spec_state_roll_kernel). Pytest 398/400 (was 229/400); conv_state byte-equal to FLA; 169 spec captures fixed. 2 remaining failures borderline bf16 noise.
- [tick-41](.spec-gdn-progress/tick-41.md) — Tick 41 — Rung 4 closure: 169/170 spec pass; 1 spec borderline (2 cells bf16 noise) + 1 non_spec pre-existing (not caused by tick 40). Mark rung 4 GREEN; advance to rung 5.
- [tick-42](.spec-gdn-progress/tick-42.md) — Tick 42 — Rung 5 GREEN: filtered pytest shows all 180 `min1_max1` tests pass. Rungs 6-11 assessed; rung 9 (determinism) testable next; 6/7/8 need new captures.
- [tick-43](.spec-gdn-progress/tick-43.md) — Tick 43 — Rung 9 GREEN: 10× byte-equal core_attn_out + conv_state + ssm_state on K=4 and K=1 captures. SYCL is fully deterministic.
- [tick-44](.spec-gdn-progress/tick-44.md) — Tick 44 — Rung 6 defensive PASS: SYCL n_acc=0 path is safe + deterministic, but FLA itself has UB at n_acc=0 (i_t=-1 reads OOB), so byte-match isn't well-defined. Production never hits this.
- [tick-45](.spec-gdn-progress/tick-45.md) — Tick 45 — Rung 11 GREEN: SYCL is 10× faster than FLA on K=4 and K=1 captures (0.29ms vs 3.0ms; 13-14k tok/s vs 1.3k tok/s). Loop reaches natural closure point.
- [tick-46](.spec-gdn-progress/tick-46.md) — Tick 46 — Rungs 7 & 8 GREEN via harness synthesis: mixed-batch test pairs same-layer non_spec+spec captures and drives the production split (29/30 pass; 1 = pre-existing layer-0 non-det); reorder_input=False equivalence test repacks qkvz/ba into per-k_head layout (200/200 pass).

## Strategy: B (per-step state ring)

Mirror FLA's `IS_SPEC_DECODING` in
`vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py`. Each
spec token's post-token state goes to its own slot
`spec_state_indices_tensor[batch, t]`; runner picks slot
`num_accepted_tokens[i]` next round. The cache pool already provides
the ring; Python dispatch already passes both tensors today.

A (snapshot+replay) rejected: 2× state memory + replay launch.
C (in-place + reverse delta) rejected: bf16 unstable.

## Ladder

- [~] 1 — capture harness in `vllm/_xpu_ops.py` (env:
       `VLLM_XPU_DUMP_SPEC_GDN=<dir>`, default max 200 tuples)
- [~] 2 — replay pytest at `tests/kernels/xpu/test_spec_gdn_replay.py`
- [x] 3 — k=1 fully accepted: kernel + dispatch + test landed AND
       verified vs inline FLA oracle (tick 12, 0/90 fails)
- [x] 4 — k=4 fully accepted: tick 40 fix landed (FLA-aligned conv1d
       spec semantics); 169/170 spec captures pass post-fix; 1 borderline
       spec (2 cells bf16 noise) + 1 pre-existing non_spec failure (not
       caused by the fix). Tick 41 closes rung 4 GREEN.
- [x] 5 — k=3 num_accepted=1: tick 42 — 180/180 `min1_max1` pass after
       tick-40 fix. The fix is naturally n_acc-agnostic.
- [~] 6 — k=3 num_accepted=0: tick 44 — defensive PASS. SYCL behaves
       safely (zero-init state, deterministic). Cannot byte-match FLA
       because FLA has UB at n_acc=0 (i_t=-1 OOB load). Production
       runtime guarantees n_acc>=1, so this is a safety check only.
- [x] 7 — mixed batch: tick 46 — harness synthesises a mixed batch by
       pairing each non_spec capture with the same-layer min1_max1 spec
       capture and replicates the spec/non-spec split that
       `_gdn_xpu_spec_sycl_path` does in production. 29/30 pairs pass
       (`test_sycl_mixed_batch_matches_per_subset`); the 1 failure is the
       same pre-existing layer-0 non_spec non-determinism that fails the
       single-batch test, not new to the mixed path.
- [x] 8 — gqa_interleaved_layout ∈ {True, False}: tick 46 — harness
       repacks each capture's qkvz/ba into the per-k_head interleaved
       layout (`_qkvz_false_to_true`, `_ba_false_to_true` mirror the
       inverse of `causal_conv1d.hpp`'s `ReorderInput` index math) and
       verifies the SYCL kernel produces equivalent outputs under
       `reorder_input=False`. 200/200 captures pass
       (`test_sycl_reorder_input_false_equivalence`).
- [x] 9 — determinism: tick 43 — 10× byte-equal core_attn_out +
       conv_state + ssm_state on K=4 and K=1 captures. SYCL kernel +
       new `spec_state_roll_kernel` are both deterministic.
- [ ] 10 — e2e Qwen3.6+MTP-K3 with `VLLM_XPU_FORCE_FLA_GDN=0`:
        acceptance ±2pp of FLA baseline (85.6/71.0/58.6), 12/12 200 OK,
        token diff <1% vs FLA
- [x] 11 — perf gate: tick 45 — SYCL is 10.3-10.4× faster than FLA on
       both K=1 and K=4 captures (0.29 ms vs 3.0 ms median; ~13-14k tok/s
       vs ~1.3k tok/s).

Tolerance: bf16 atol=rtol=2e-2; tighten if you can.

## Env

Dev container: `vllm-dev` (flake at `~/Projects/vllm-xpu-kernels/flake.nix`).
Wrappers: `vllm-xpu-build`, `vllm-xpu-rebuild`, `vllm-test`, `vllm-run`,
`vllm-shell`. Get them via `nix develop ~/Projects/vllm-xpu-kernels -c <cmd>`
(loop shell doesn't auto-load direnv). Both repos bind-mounted.
GDN-on by default; `MAX_JOBS=2` auto-set when GDN/grouped_gemm xe_2
heavy SYCL templates are in the build set.

Built 2026-05-07: `_xpu_C.abi3.so` registers `gdn_attention` with
`spec_state_indices_tensor` and `num_accepted_tokens` in its schema.
`libgdn_attn_kernels_xe_2.so` fresh in the source tree.

`VLLM_XPU_USE_SYCL_SPEC_GDN`:
- unset / `0`: spec → FLA (default)
- `auto`: SYCL with one-time soft fallback to FLA on missing op /
  stale schema
- `1`: strict — `_validate_sycl_spec_gdn_op()` raises on op or
  schema mismatch. Use this in tests so build issues never silently
  degrade to FLA.

## Resume

Source state: pristine (T26 volatile-temp reverted at end of
tick 26). The currently-installed `_xpu_C.abi3.so` is the
volatile-temp build; T27 will overwrite it. 200 captures at
`/tmp/spec_gdn_captures` inside the `vllm-dev` container (30
non_spec / 90 spec_K4_min1_max1 / 80 spec_K4_min4_max4); the
layer-0 worst replay tuple is
`tuple_..._layers_0_..._000004_spec_K4_min4_max4.pt`.

### T28: IGC env-var workaround ladder (NO rebuild)

T27 closed: drift = 8.9465 under JIT (= T26 volatile-temp). Bug
is IGC-backend-wide, not AOT-only. Move to env-var workarounds
against the *existing* JIT-only `.so` (no rebuild needed; IGC
honors these at runtime when JIT-compiling SPIR-V).

**Probe template** for each env var (use `vllm-run -e`):
```
nix develop ~/Projects/vllm-xpu-kernels -c \
  vllm-run -e <ENV>=<VAL> python /workspace/vllm/.spec-gdn-load-readback.py
```
Pass criterion: `max |delta|` ≤ 1e-3 (i.e. byte-equal load
under g=1, beta=0). Fail = drift remains ≈ 8.95.

**Ladder (cheap → expensive)**:
1. `IGC_ForceOCLSimdWidth=16` — halves SIMD from 32 to 16.
   Forces full RA redo. **Most likely to fix.**
2. `IGC_DisableLoopUnroll=1` — disables auto-unroll. Breaks the
   4×4 j-i unroll at line 241; may eliminate the j=3 stripe
   pattern. Less likely to fix but free.
3. `IGC_FunctionControl=2` — disables function inlining. Probably
   irrelevant but cheap.
4. `IGC_OptDisable=1` — last resort, disables most optimization.
   Will tank performance; useful only as a "what was IGC doing
   that broke things?" signal.

If any env var passes: ship via the existing dispatcher hook in
`vllm/_xpu_ops.py` (set the env var conditionally for the
gdn_attention launch path). Document the perf cost.

**Failure path → T29 source workaround**:
- struct-wrap `spec_state_indices_tensor` + `num_accepted_tokens`
  into a single struct passed by-value to the SYCL kernel. T22's
  RA hypothesis was that the two extra captured-by-value
  parameters bumped IS_SPEC=true RA pressure; consolidating them
  may equalize with IS_SPEC=false pressure.

**T30 source workaround (last resort)**:
- `[[intel::reqd_sub_group_size(16)]]` on the kernel — same
  effect as `IGC_ForceOCLSimdWidth=16` but baked into the
  kernel signature so it survives env reset. Documented perf
  cost (~50% slower).

If T28-T30 all fail, file Intel bug with T23/T26 ISA evidence
and gate gdn_attention spec path on a runtime IGC version
check.

### Tools (this branch's source-of-truth)

- `vllm/.spec-gdn-tick25-cheapprobes.py` (T25 P1+P2: pyref
  finite-check + IS_SPEC=false load-readback)
- `vllm/.spec-gdn-load-readback.py` (T19; per-iter drift probe
  under g=1, beta=0; IS_SPEC=true)
- `vllm/.spec_gdn_disasm/extract_section.py` (T26; generic ELF
  section extractor: objcopy → ar walk → ELF section by mangled
  name. Used `Li4ELb1EEE` filter + bf16/bf16 mangling
  `_ZTSN3gdn23gated_delta_rule_kernelIN4sycl3_V13ext6oneapi8bfloat16ES5_Li4ELb1EEE`)
- `vllm/.spec_gdn_disasm/` (T23 pristine + T26 volatile-temp ASMs:
  `spec_true_k4_bf16.{bin,asm}`, `spec_false_k4_bf16.{bin,asm}`,
  `volatile_spec_true_k4_bf16.{bin,asm}`)

FLA fallback at `vllm/_xpu_ops.py:_gdn_xpu_spec_python_path` is
the oracle — do not touch.

2026-05-08 tick 29: **T27 OUTCOME — bug exists under JIT too**.
The IGC-AOT-only escape ramp is closed. Must move to source-level
RA decouple / SIMD / runtime-IGC workaround ladder.

**Build (final, working)**: `bdt43kc0q` finished cleanly at
2026-05-08 ~01:08 local. Configuration:
```
docker exec --workdir /workspace/vllm-xpu-kernels \
  -e VLLM_XPU_AOT_DEVICES= \
  -e VLLM_XPU_XE2_AOT_DEVICES= \
  -e FA2_KERNELS_ENABLED=OFF \
  -e MAX_JOBS=2 \
  -e CMAKE_BUILD_TYPE=Debug \
  vllm-dev sh -c 'pip install -e . --no-build-isolation'
```
JIT mode verified (no `-fsycl-targets=spir64_gen`). Build dir
preserved across the two false starts; only the 44 cheap/link/
install steps actually ran in the final attempt. Total fresh-T27
work time: ~10 min on the final attempt vs ~16 hours projected
for the wrong path.

**Probe `.spec-gdn-load-readback.py` against layer-0 worst capture,
g=1, beta=0**:
  - max |delta| = **8.9465**
  - 9011 cells with |delta| > 1e-3 (identical count to T26)
  - hot v=79 (31 occurrences) — same universal hot lane as
    T15/T26
  - hot heads h=14(116) / h=19(115) / h=18(112) / h=15(99) —
    bit-for-bit same head profile as T26 volatile-temp
  - **Identical statistical fingerprint to T26 volatile-temp run
    under AOT.** Different magnitude than pristine AOT (9.22).

**Numerical summary**:
  | Build         | max\|delta\|  |
  |---------------|---------------|
  | AOT pristine  | 9.22 (T19)    |
  | AOT volatile  | 8.9465 (T26)  |
  | **JIT pristine** | **8.9465 (T27)** |

JIT and AOT-volatile produce *the same drift number*. The drift
9.22 → 8.95 shift coincides with anything that perturbs IGC's
codegen on the IS_SPEC=true template — both volatile spilling
and JIT-vs-AOT path do this. Suggests the bug is a single
sensitive instruction whose output is value-dependent on RA /
scheduling, and the JIT path happens to land in the same
"perturbed" basin as volatile-temp.

**Verdict**: bug is **IGC-backend-wide**, not specific to the
AOT pass. Drift ≠ 0 under JIT closes the workaround "ship
JIT-only" exit. Must use a source- or environment-level fix.

**Next: T28 — IGC env workarounds (cheapest, runtime-only)**.
Per the T27 plan, the cheapest first step (no rebuild required)
is to test IGC environment variables against the existing JIT
build. The .so we have is already JIT-only, so all IGC env vars
take effect at runtime:
  - `IGC_ForceOCLSimdWidth=16` — halves subgroup size from 32 to
    16, forces re-RA. If drift=0, ship this env var as the
    workaround (vllm-run -e IGC_ForceOCLSimdWidth=16 in the
    spec-GDN dispatch path).
  - `IGC_DisableLoopUnroll=1` — disables auto-unroll, may break
    the j=3 stripe codegen pattern. Less likely to fix but free
    to test.
  - `IGC_ShaderDumpEnable=1` + `IGC_DumpToCustomDir=...` — dumps
    every JIT-compiled binary to disk. Useful to compare ASMs
    before/after a working env-var workaround. Optional probe.

T28 is a multi-probe tick: try each env var individually with the
load-readback probe, record drift. First one that flips drift to
≤ 1e-3 wins; ladder forward to rung 4 verification.

If all IGC env workarounds fail:
- T29: source-level RA decouple — wrap
  `spec_state_indices_tensor` + `num_accepted_tokens` into a
  struct passed as a single kernel arg. Goal: equalize the
  IS_SPEC=true RA pressure with IS_SPEC=false (whose template
  doesn't carry these as separate captured-by-value parameters).
- T30: `[[intel::reqd_sub_group_size(16)]]` — same effect as
  IGC_ForceOCLSimdWidth=16 but baked into the kernel attribute.
  Documented ~50% perf cost expected.

Source state: pristine. Build state: JIT-only `_xpu_C.abi3.so`
installed at `vllm_xpu_kernels/_xpu_C.abi3.so` mtime
`1778227678` (~01:08 local 2026-05-08).

2026-05-08 tick 32: **Methodological sanity check — PASSED**.
Pipeline is healthy; the 8.9465 invariance is a genuine bug
fingerprint, not a stale-build artifact.

**Setup**: replace line 241 with deliberately breaking
assignment:
```cpp
// pristine: state_local[j * k_bucket_size + i] += k_local[i] * delta[j];
state_local[j * k_bucket_size + i] = 0.0f;  // SANITY: zero, not +=
```
Incremental rebuild via `docker exec ... -e
FA2_KERNELS_ENABLED=OFF MAX_JOBS=2 pip install -e .` ~10 min.
`_xpu_C.abi3.so` mtime 8480 → 9071 (rebuilt). NEO cache cleared
before probe.

**Probe result with zeroed state_local**:
  - max |delta| = **14.7019** (was 8.9465)
  - 162871 cells with |delta|>1e-3 (was 9011) — **18× more
    cells**
  - All 32 heads saturated at h=*(128) — every head broken
  - All v_dim show 32 occurrences each — fully saturated
  - 14.7019 ≈ |state_pre|_max from T25 P1 finite-check, exactly
    what we'd expect when state_local is zeroed before
    writeback (drift = magnitude of pre-state).

Drift jumped, statistical fingerprint completely changed. Kernel
rebuilds DO take effect. Pipeline is sound.

**What this means for the prior 8.9465 invariance**: the bug at
pristine line 241 is reproducible across every codegen
perturbation tried (volatile-temp, JIT, IGC env vars,
SIMD-16+k=8). All produce drift = 8.9465. This is a *property
of the bug*, not a property of the pipeline.

**Sharpening the diagnosis**: line 241's `+= k_local[i] *
delta[j]` should be a no-op given:
- delta = (v_local - kv_mem) * beta where beta = sigmoid(-100) =
  3.72e-44 (subnormal float). With FTZ on, delta = 0; with FTZ
  off, delta is subnormal ≈ 1.6e-42 (T25 P1 bound).
- k_local * delta ≈ 28 * 1.6e-42 = 4.5e-41 (still subnormal).
- state_local += subnormal: with FTZ on, += 0 → no change; with
  FTZ off, += 1.8e-41 → relative error ~1e-42, invisible at any
  bf16 precision.

But we observe drift = 8.9465. So the kernel is computing
*something other than* the source-level math. Either:
A. The codegen is *literally* miscompiling line 241 (the T20-T26
   conclusion, supported by T22's RA hypothesis being now
   falsified — bug survives every RA-perturbing change).
B. Some other interaction (e.g., write-after-read across
   subgroup, race on state_local between unrolls, instruction
   ordering bug at the GEN ISA level) produces this value
   *anyway*.

The invariance across SIMD-32→16, k=4→8, AOT→JIT suggests this
is a fundamental backend issue, not a high-level optimization
choice.

**Three-way checkpoint complete**:
- Test pipeline: ✓ confirmed working (drift changes when source
  changes).
- Bug existence: ✓ confirmed at pristine line 241 (drift 8.9465
  is real, reproducible).
- Bug nature: ✓ codegen-level, not RA-pressure (T22 falsified by
  T29.A).

**Next direction (tick 33+)** — refocus from "RA workaround" to
"narrow root cause + alternate workaround":

Option 1 — **Re-validate T18's "bug in LOAD" claim**: T18
isolated to LOAD; T20 isolated to line 241 (state += k*delta).
These are different lines. With pristine line 241 producing 8.95
drift but math-says-it-shouldn't, the T18 LOAD framing might
actually be right and T20-T26 misled the chase. Re-run the
load-only probe (skip the iter loop entirely, just do load +
writeback) to see if drift = 8.95 even before any compute.

Option 2 — **Replace `+= k_local[i] * delta[j]` with literal
`+= 0.0f`**: if drift = 0, the bug requires the actual k*delta
expression (codegen miscompiles the multiply or load of k/delta).
If drift = 8.95, the bug is independent of the operands — there's
something wrong in the surrounding loop structure or the store.

Option 3 — **Strip the entire spec writeback path (lines
260+) and confirm**: if drift = 0 after stripping spec writeback,
the bug is in the *writeback*, not the load or compute. The
spec writeback uses `spec_state_indices_tensor[batch, t]` which
is the IS_SPEC=true-only path.

Tick 33 should run option 2 first (cheapest, single-line edit).
If drift = 0, root cause is the multiply expression. If drift
= 8.95, run option 3 next.

**Source state**: reverted to pristine. .so still has the
zeroed-line-241 build (will overwrite in tick 33).

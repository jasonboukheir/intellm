2026-05-07 tick 21: source-level workarounds all bit-identical to
buggy original. Compiler folds them.

Tried three source-level rewrites of line 241 to break the codegen:

1. **Temp variable** (separate mul + add):
```cpp
const float prod = k_local[i] * delta[j];
state_local[j * k_bucket_size + i] += prod;
```
Drift: 5.7994 / 7.6807 / 9.2231 (BIT-IDENTICAL to original).

2. **Loop split** (state-update loop separate from res loop):
```cpp
for (j) for (i) state_local[j*K+i] += k_local[i] * delta[j];
for (j) for (i) res[j] += state_local[j*K+i] * q_local[i];
```
Drift: BIT-IDENTICAL.

3. **Volatile temp** (force memory store/load between mul and add):
```cpp
volatile float prod = k_local[i] * delta[j];
state_local[j * k_bucket_size + i] += prod;
```
Drift: BIT-IDENTICAL.

All three produce numerically identical output to the original,
including the same drift bug. The compiler is folding all source forms
back to the same generated code at the IR/SPIR-V level. Source-level
restructuring is insufficient.

**Implication**: the bug is NOT something the SYCL/C++ compiler is
choosing differently per source form — it's emitting one specific
SPIR-V/Xe binary pattern for "state_local[idx] += <runtime_value>"
across all source variants, and that pattern is buggy.

Tick 21 also confirms tick 20's narrowing: even with `prod` forced
through volatile memory (so the multiplication is genuinely separate
from the add), drift is unchanged. The bug is in the ADD side, not
the multiply side: `state_local[idx] += <any_runtime_float>` corrupts
state_local across iterations even when the runtime float is zero.

**Hypothesis revision**: the compiler's lowering of `state_local[idx]
+= X` (for any X) emits an instruction sequence that has a hardware
or codegen quirk on Intel Xe at certain register allocations. The
literal `+= 0.0f` from tick 20 is the only form that bypasses this,
because the compiler statically proves the assignment is a no-op
and elides it entirely.

Tick 22 plan: dump the actual generated code to see what's emitted.

Approach: rebuild with `-save-temps` on the SYCL device-code pass to
get LLVM IR / SPIR-V for the `gdn_attention_kernel<bf16, IS_SPEC=true,
k_bucket_size=4>` instantiation. Inspect the FMA / mad / store pattern
emitted for line 241 vs lines 210, 211. Look for:
- Register pressure causing spills.
- FMA with denormal mode quirks.
- Unintended reload of state_local from a wrong stack offset.

Search for the inner loop body in the IR; correlate the j=3 (v=79)
hot-spot with specific SPIR-V instructions.

If IR inspection reveals a clear codegen pattern that's buggy, tick 23
proposes a fix (compiler flag, source pragma, or restructuring that
the compiler can't fold).

Source state: pristine (all experiments reverted). .so rebuilt.


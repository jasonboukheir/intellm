2026-05-07 tick 22: more workarounds tried, all bit-identical.
Self-assign confirms compiler elides provably-no-op writes.

Two more attempts:

1. **`#pragma clang fp contract(off)`** in a compound block around the
   inner loops (to disable FMA fusion). After fixing the first attempt
   (pragma must be at start of compound stmt, not in the middle of
   code), build succeeded. Drift: 5.7994 / 7.6807 / 9.2231 — BIT-IDENTICAL
   to original. Either the pragma was applied but had no effect (FMA
   isn't the bug), or icpx silently dropped the pragma at SYCL device
   compilation.

2. **Self-assign** `state_local[idx] = state_local[idx]` (replacing
   the buggy line 241 with a pure no-op write):
```cpp
state_local[j * k_bucket_size + i] = state_local[j * k_bucket_size + i];
```
   Drift: 0.0 across all iter pairs.

The self-assign result (drift = 0) confirms the compiler ELIDES writes
that it can prove are no-ops at the IR level. This explains why
`+= 0.0f` (literal zero) and self-assign both give drift 0 — both are
provably no-op stores and the compiler removes them.

But ANY runtime-computed value, even one that is mathematically zero
(`k_local * delta` with delta=0), gets a real STORE instruction
emitted. That store is what corrupts state_local across iterations.

**Cumulative bisection table** (g=1, beta=0 input, iter0-vs-iter3 max):

  Source variant (line 241)                                    drift
  pristine (`+= k_local[i] * delta[j]`)                          9.22
  commented out                                                  0.00
  `+= 0.0f`                                                      0.00
  `state_local[idx] = state_local[idx]`                          0.00
  `const float p = k * delta; state_local += p;`                 9.22
  loop split (state-update loop separate from res loop)          9.22
  `volatile float p = k * delta; state_local += p;`              9.22
  `volatile float state_local[16]`                               9.22
  `#pragma clang fp contract(off)` around inner loops            9.22

Pattern: drift = 0 IFF compiler can statically prove the store is
no-op AND elide it. ANY emitted store with runtime-computed value
triggers the bug. Source-level workarounds are all canonicalized to
the same machine code.

**Refined hypothesis**: the bug is in the IGC (Intel GPU Compiler)
SPIR-V → Xe ISA lowering, not in icpx's SYCL device code generation.
icpx produces consistent SPIR-V regardless of source form (its IR
canonicalizer normalizes them). IGC then lowers that SPIR-V to ISA
with a buggy instruction sequence at line 241's specific pattern.
Source-level pragmas in SYCL don't reach IGC.

This is consistent with "v=79 dominates" being a per-(j-lane,
register-allocation) bug at the ISA level — the kind of thing that
manifests at specific GRF allocations, not at IR.

**Tick 23 plan**: dump device-side SPIR-V from the built .so and
inspect the emitted code for the spec-true template instantiation.

Approach:
1. Use `ocloc disasm` or `spirv-dis` to disassemble the .spv blob
   embedded in `libgdn_attn_kernels_xe_2.so`.
2. Find `gdn_attention_kernel<bf16, IS_SPEC=true, k_bucket_size=4>`
   and locate the inner loop body.
3. Inspect the FMA / `mad` / store sequence emitted for line 241.
4. Compare against the same instantiation with line 241 commented
   out (which has drift = 0). The diff localizes the buggy
   instruction(s).

If the diff reveals a clear pattern (e.g., a specific `mad`/`mul`
+ `mov`/`store` sequence at j=3), tick 24 tries an IGC-specific
workaround (e.g., `-Xs "-options -cl-no-subgroup-ifp"` or similar)
or a code restructuring that produces a different SPIR-V.

Alternative tick 23 (if SPIR-V tooling isn't available in container):
test JIT vs AOT — remove `-fsycl-targets=spir64_gen` to get JIT
SPIR-V, see if drift differs. If JIT gives drift 0, the bug is purely
AOT (IGC).

Tools used: 3 incremental rebuilds, 1 build error fix. Source state:
pristine.


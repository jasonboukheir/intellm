2026-05-07 tick 20: bug isolated to a SINGLE source statement (line 241).

Bisected via incremental rebuilds with the load-readback probe (g=1,
beta=0). All edits to `csrc/xpu/gdn_attn/gated_delta_rule.hpp`. Same
layer-0 worst capture, same probe, same expected output (drift = 0):

  Source variant                                    iter0vs3 max | hot(h,v)
  pristine (lines 210, 241 active)                  9.22         | 1760
  `volatile float state_local[16]`                  9.22         | 1760  (bit-identical)
  L210 commented + L241 commented                   0.00         | 0     (clean)
  L210 active   + L241 commented                    0.00         | 0     (*= innocent)
  L210 active   + L241 active                       9.22         | 1760  (drift returns)
  L210 active   + L241 → `state_local += 0.0f`      0.00         | 0     (literal elided)

**Verdict**: bug is exclusively in
```
state_local[j * k_bucket_size + i] += k_local[i] * delta[j];
```
when the compiler emits the runtime mul+add. With delta=0 at runtime
(beta=0 → delta = (v - kv_mem) * 0 = 0), the math is +=0 → no-op.
The compiled instruction is NOT a no-op — it shifts state_local by
~2 units per iter. By iter 3, max cell drift is 9.22 across all
heads/v-positions where state magnitudes are large.

**Rejected hypotheses**:
- Register reuse / scheduler aliasing: volatile state_local made
  zero difference (bit-identical output). Compiler wasn't using
  cached registers in the buggy way.
- Memory aliasing between kv_mem and state_local: same volatile
  test rules out shared storage.
- `*= g` codegen: commenting out 241 alone (with 210 active) → 0.0
  drift. The decay multiply is exact when g=1.
- LOAD codegen: same commented-out test → state_post[iter t] equals
  pre-state at load slot exactly, for every t. The load reads
  ssm_state correctly.

**Suspect codegen pattern**: SYCL/SPIR-V (Intel oneAPI 2025.3) likely
emits `state_local[idx] = sycl::fma(k_local[i], delta[j], state_local[idx])`,
and that FMA — when `delta[j] = 0` at runtime — is corrupting
state_local. Could be:
- An FMA instruction with denormal-handling that misbehaves on the
  zero multiplicand case at certain register allocations.
- A vectorized FMA across the j unroll that mis-routes lanes.
- A scheduler quirk that interleaves with line 243 (`res += state * q`)
  in a way that mutates state_local before res reads it.

**Note on REAL inputs**: with real (g≈0.998, beta≈0.93, delta non-zero)
the same statement runs but with a non-zero delta. The visible
divergence in `core_attn_out` (max 1.22 at v=79, tick 11) is the same
codegen issue scaled by `|k_local * delta|` magnitude. v=79 dominates
in layer 0 because layer 0's state magnitudes at v=79 are largest.
The "value-sensitive amplitude" pattern from tick 12 falls out of
this naturally.

Source is back to pristine state after experiments; .so rebuilt.

**Tick 21 plan**: codegen workaround. Cheapest first — break the FMA
pattern by splitting the expression across two statements:
```cpp
const float prod = k_local[i] * delta[j];  // separate mul to a temp
state_local[j * k_bucket_size + i] += prod;  // separate add
```
or better, decouple the state update from the res accumulation by
splitting the j/i loop:
```cpp
// Loop A: state update only
for (int j ...) for (int i ...) {
  state_local[j*K+i] += k_local[i] * delta[j];
}
// Loop B: res accumulation only
for (int j ...) for (int i ...) {
  res[j] += state_local[j*K+i] * q_local[i];
}
```
The second variant is ALSO valid math (state_local writes finish
before reads start) and may produce different codegen since the
compiler can schedule the loops independently. If either fix drops
the tick-12 layer-0 v=79 baseline diff from 1.22 → bf16 noise, ship
it.

If neither helps, tick 22 dumps SPIR-V/LLVM IR for the spec-true
template instantiation around the buggy statement and inspects the
emitted FMA.

Tools used: edits to `csrc/xpu/gdn_attn/gated_delta_rule.hpp` (all
reverted post-experiment); 4 incremental rebuilds (~30s each, cmake
cache survived); 4 reuses of `vllm/.spec-gdn-load-readback.py`.


2026-05-08 tick 34: **ROOT CAUSE — `act_sigmoid(-100)` returns
non-zero on device under fast-math. The line-241 "codegen bug"
framing from T17-T26 was a misdiagnosis.**

**The decisive probe chain in tick 33-34**:

| Source change                                   | drift  | cells   |
|-------------------------------------------------|--------|---------|
| Pristine `+= k_local[i] * delta[j]`             | 8.9465 | 9011    |
| Sanity: `= 0.0f` (no +=, force zero)            | 14.7019| 162871  |
| `+= 0.0f` literal                               | 0.0000 | 0       |
| `+= delta[j]` (no multiply)                     | 11.5583| 186880  |
| `+= delta[j]` + force `float beta = 0.0f;`      | **0.0000** | **0** |

**Conclusion**: when `beta = 0` is forced (bypassing
`act_sigmoid(b)`), drift drops to zero. So `act_sigmoid(-100)`
on device is NOT returning 0 — it's returning some value large
enough to make `delta = (v - kv_mem) * beta` non-trivial.

**Why act_sigmoid breaks at b=-100**:
```cpp
static inline float act_sigmoid(float& x) {
  return 1.0f / (1.0f + sycl::exp(-x));
}
```
With `x = -100`: `sycl::exp(100)` mathematically ≈ 2.69e43,
which **overflows fp32 max (3.4e38)**. IEEE math: returns +inf
→ `1/(1+inf) = 0`. But on device under SYCL's default math
flags (`-fapprox-func -funsafe-math-optimizations
-fno-signed-zeros -mreassociate -freciprocal-math
-ffp-contract=fast-honor-pragmas`), the overflow path is
optimized differently. Likely IGC uses the algebraic
transformation `1/(1+exp(-x)) = exp(x)/(1+exp(x))` for fast
evaluation:
  - `exp(-100)` = 3.72e-44 (subnormal)
  - `exp(-100) / (1 + exp(-100))` = 3.72e-44 / 1 = 3.72e-44
    (subnormal)
  - With `-fdenormal-fp-math=preserve-sign`, this should flush
    to ±0 — but the actual value reaching `delta = (v-kv) * beta`
    must be much larger to produce drift = 11.56.

Possible: IGC's range-reduced exp returns a non-trivial value
for x=100 (not inf, not zero), or the fast-math-rewritten
sigmoid takes a path that bypasses the FTZ handling.

Empirical evidence is enough: forcing `beta = 0.0f` fixes drift
completely.

**What this invalidates**:
- T17-T26 narrowing: "bug at gated_delta_rule.hpp:241". WRONG.
  The bug is at line 168 (`act_sigmoid` call) — and only
  expresses when b is large negative.
- T22 RA-pressure hypothesis: irrelevant. Bug isn't in line 241
  at all.
- T23-T24 ASM diff hunt: chasing a phantom — the binaries are
  fine, the input to line 241 (delta) is wrong upstream.
- T26 volatile-temp ASM diff: irrelevant for the same reason.
- T27 JIT-vs-AOT: bug is the same in both because `act_sigmoid`
  has the same issue regardless of compile mode.
- T28 IGC env-var ladder: drift identical because act_sigmoid's
  mishandling is independent of these env vars.
- T29.A SIMD-16: same — sigmoid overflow doesn't depend on RA.

**Critical scope question**: did the rung 4 RED actually fail
because of `act_sigmoid(-100)`?

The synthetic probe `vllm/.spec-gdn-load-readback.py` *forces*
`b = -100`. That's NOT what real production data looks like —
real `b` values come from logits, typically in roughly
`[-10, 10]`. For those values, `act_sigmoid` is well-behaved
(no exp overflow), and the bug should not manifest.

But T5's original rung 4 failure was on real data
(`/tmp/spec_gdn_captures` from a Qwen3.6 + MTP-K3 capture).
That failure showed v=79/107 hot — coincidentally the same
v=79 the synthetic probe shows. That coincidence may be a
property of the data layout (some v dim is always more
sensitive), not a shared root cause.

**Tick 35 plan — verify on real data**:
1. Revert all source changes (already done end of tick 34).
2. Rebuild pristine.
3. Run **rung 4 directly**: `tests/kernels/xpu/test_spec_gdn_replay.py`
   on the layer-0 worst capture, NO synthetic b override.
4. Observe: does pristine fail? If yes, what's the diff
   pattern?

If rung 4 STILL fails on pristine real data, the bug is
something else (not act_sigmoid overflow). Investigate from
scratch — but with the new framing that line 241 is fine.

If rung 4 PASSES on pristine real data, the synthetic probe
was the only thing failing. The "bug" was a probe artifact;
rung 4 was actually green all along under the relaxed
tolerance. Re-verify rung 4 status with stricter tolerance
once `act_sigmoid` is patched (any production-class b cap
won't trigger sigmoid overflow, but a defensive saturation in
the kernel is still good hygiene).

**Source state**: reverted to pristine.

2026-05-08 tick 33: **MAJOR FINDING — drift = 0.0 with `+= 0.0f`
literal**. Bug is operand-dependent.

**Setup**: edited line 241 from
`state_local[j * k_bucket_size + i] += k_local[i] * delta[j];`
to
`state_local[j * k_bucket_size + i] += 0.0f;  // PROBE: literal 0`.
Incremental rebuild ~7 min. `_xpu_C.abi3.so` mtime 9071 → 9505.
NEO cache cleared.

**Probe result**:
  - max |delta| = **0.0000**
  - 0 cells with |delta|>1e-3
  - **Bug fully suppressed.** state_local pre == post,
    byte-equal under `g=1, beta=0` (compute is now strict identity).

**What this rules in / out**:
  - **Loop structure is fine**. The same 16-iter unroll, same
    store target, same surrounding code → drift = 0 when the
    RHS is a literal. So loop, store, indexing, address
    arithmetic are all correct.
  - **Bug is in the multiply or the operands**, not in the `+=`
    or the address computation.

**Mathematical expectation vs reality**:
Per T25 P1: `delta = (v - kv_mem) * sigmoid(-100) =
(v - kv_mem) * 3.72e-44`. With v - kv_mem ≤ ~17, |delta| ≤
6.3e-43 (subnormal). Then `k_local * delta ≤ 28 * 6.3e-43 ≈
1.8e-41` (subnormal). Then `state += 1.8e-41` should be a no-op
on any IEEE-conformant float (relative error ≈ 1e-42, far below
fp32 ulp).

But device produces drift = 8.9465. So device math != IEEE
math somewhere along this chain.

**Suspect chain — IGC's aggressive fast-math flags**:
The build uses (per `compile_commands.json`):
  - `-O3 -DNDEBUG`
  - `-fapprox-func` (allow approximate transcendentals)
  - `-funsafe-math-optimizations` (allow non-IEEE optimizations)
  - `-fno-signed-zeros`
  - `-mreassociate` (allow reorder)
  - `-freciprocal-math`
  - `-ffp-contract=fast-honor-pragmas`

These can mishandle subnormals. In particular:
  - `-funsafe-math-optimizations` typically implies "denormals
    flushed to zero" in IGC, but NOT necessarily before they're
    used in a multiply. If `delta` is computed and stays
    subnormal, then `k * delta` runs at low SIMD throughput AND
    may produce wrong values on Xe2 (specific HW limitation:
    subnormal multiply latency is reportedly 8x slower and uses
    a microcode path that has known issues on some IGC revs).
  - On a buggy IGC revision, the subnormal multiply might
    return e.g. `k` instead of `k*subnormal`, or some uninitialized
    register value, producing the observed 8.9465 drift.

**Tick 34 — narrow operand**: replace line 241 with
`state_local[idx] += delta[j]` (delta alone, no multiply).
  - drift = 0 ⇒ multiply is the buggy operation; delta is fine
    on its own (consistent with FTZ working at the load/store
    boundary but failing inside the multiply).
  - drift > 0 ⇒ delta itself is somehow non-zero on device
    (subnormal handling at the front-end, or dt/beta computation
    produces large delta despite source-level math saying it
    should be tiny).

[Build for this probe kicked off at end of tick 33; result
arrives in tick 34.]

**Source state**: edited (`+= delta[j]` for tick 34).
.so: literal `+= 0.0f` build (will be overwritten by tick 34
build before probe).

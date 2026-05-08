2026-05-07 tick 26: T26 volatile-temp ASM diff INCONCLUSIVE. The
"RA effectively unchanged" premise from T21 was wrong at the IGC
level — `volatile float prod` triggers a heavy memory spill that
reshuffles RA across the whole kernel.

**Setup**: edit `gated_delta_rule.hpp:241` to:
```cpp
volatile float prod = k_local[i] * delta[j];
state_local[j * k_bucket_size + i] += prod;
```
Incremental rebuild (~270s, vllm-xpu-rebuild) succeeded; .so refreshed.

**Step 3 sanity (drift probe)**: `.spec-gdn-load-readback.py` against
the layer-0 worst capture, g=1, beta=0:
  - max |delta| = **8.95** (vs pristine 9.22)
  - 9011 cells with |delta|>1e-3 (similar count)
  - hot v=79 (31 head/batch occurrences) — same universal hot lane
  - hot heads h=14(116) / h=19(115) / h=18(112) / h=15(99) — same as
    pristine bug shape

So bug **persists** with similar magnitude and identical statistical
shape. Premise "drift==pristine" effectively holds (8.95 ≈ 9.22 in
bf16 atol terms).

**Step 4 ASM extraction**: bf16/bf16 k=4 IS_SPEC=true section is
`.text._ZTSN3gdn23gated_delta_rule_kernelIN4sycl3_V13ext6oneapi8bfloat16ES5_Li4ELb1EEE`
in ar11.
  - Pristine: 13504 bytes, 1056 iga64 lines (saved as
    `spec_true_k4_bf16.bin`/`.asm`).
  - Volatile-temp: **14272 bytes, 1102 lines (+768 bytes / +46 lines)**.
  - Saved as `volatile_spec_true_k4_bf16.bin`/`.asm`.

**Step 5 ASM diff**: dominated by RA-wide reshuffling, NOT a
localized codegen perturbation.
  - Raw `diff -u`: 2098 differing lines (out of 1102 total).
  - After register-normalize (`r\d+`→`rN`, `acc\d+`→`accN`,
    `L\d+`→`LN`): still **1914 differing lines**. ~87% of the file
    differs even after RA name-collapsing.
  - Inner-loop region (lines ~490-811 pristine, ~490-855 volatile):
    - `mad` count: **52 in both** — bug is NOT a missing/extra FMA.
    - `add` 91/90, `mul` 68/70, `mov` 55/56, `shl` 29/30, `mad` 52/52
      — op mix essentially preserved.
    - `send.ugm`: **9 → 41 (+32 ops)** in the volatile span. This
      is the "smoking gun" for why RA shuffled: `volatile float
      prod` spills to memory unconditionally on every iteration of
      the 4×4 unroll → 16 stores + 16 loads = 32 new memory ops in
      the inner loop, bumping pressure and forcing re-allocation
      throughout.

**Premise failure**: T21 reported "drift == pristine 9.22" for the
volatile-temp variant and concluded "RA effectively unchanged".
That conclusion was based on observable drift behavior, not ISA.
At the ISA level, IGC re-emits the entire kernel with different
register assignments because of the spill traffic. The volatile
keyword forces conformance with C/C++ "external observability"
which IGC implements as memory writes — heavy.

This is the same RA-wide confound that wrecked T24's spec=true vs
spec=false diff. Cannot use this experiment to localize line-241
codegen.

**Decision**: skip the controlled-ASM-diff branch entirely. Move
to **T27 (JIT vs AOT)**, which is a *binary* signal:
  - Drop `-fsycl-targets=spir64_gen` via
    `VLLM_XPU_AOT_DEVICES="" VLLM_XPU_XE2_AOT_DEVICES=""
    vllm-xpu-build`. Offload bundle ships SPIR-V only; runtime IGC
    JIT-compiles.
  - Re-run `.spec-gdn-load-readback.py`.
  - **drift=0** ⇒ bug is purely IGC-AOT. Workaround: ship JIT-only
    on Xe2 + file Intel bug with the T23 ISA evidence already in
    hand. This is the cleanest exit.
  - **drift≠0** (≈9.22) ⇒ same IGC backend regardless of mode.
    Then move to source-level RA decouple (passing
    spec_state_indices via a struct to make IS_SPEC=true RA pressure
    match IS_SPEC=false), or `[[intel::reqd_sub_group_size(16)]]`.

T27 is a full SYCL AOT rebuild (~1h45m per the build-cache feedback
memory) — kick off in the next tick, schedule wakeup near the
build's expected end, and the tick after handles probe + decide.

**Source state**: pristine (volatile-temp reverted at end of tick).
.so is the volatile-temp build (still has the spill); next tick's
T27 build will overwrite anyway.

**Tools added this tick**:
- `vllm/.spec_gdn_disasm/extract_section.py` — generic ELF
  section extractor (objcopy → ar walk → ELF section by mangled
  name). Reusable for T27 and any future ASM-bisect ticks.
- `vllm/.spec_gdn_disasm/volatile_run/` — kept the extracted
  archives + bin (`volatile_spec_true_k4_bf16.{bin,asm}`) for
  reference; do not commit.

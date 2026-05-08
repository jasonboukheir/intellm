2026-05-07 tick 25: REFRAME. T25 self-assign plan rejected; cheap
probes ran instead. Bug is **IS_SPEC=true template-specific**.

**Reviewer reframe** (rejecting the planned tick 25):
- Self-assign at line 241 is provably no-op and IGC elides it (per
  T22's drift=0 row). Diffing self-assign ASM vs pristine spec=true
  doesn't isolate the bad store — it triggers a fresh RA pass over
  the whole inner loop because the live range disappears. Same RA-
  wide confound as T24's spec_true vs spec_false diff.
- Better controlled experiment (queued for later): rebuild with the
  T21 volatile-temp variant (`volatile float prod = k_local[i] *
  delta[j]; state_local[idx] += prod;`) — drift = 9.22, store still
  emitted, RA effectively unchanged per T21. Diffing volatile-temp
  ASM vs pristine spec=true ASM isolates ONLY the line-241 codegen
  perturbation, not RA-wide.
- Skipped cheap experiments (no rebuild): pyref delta finite-check;
  IS_SPEC=false load-readback under same g=1, beta=0; JIT vs AOT;
  SIMD-16 forcing.
- Sharpened source locus: line 210 `state_local *= g` is innocent
  per T20, line 241 `state_local += k_local[i] * delta[j]` is broken.
  The `*=` is per-element scalar broadcast across the j-unroll; the
  `+=` has a different value per i AND per j → non-broadcast FMA
  chain. Bug is in the non-broadcast FMA-with-store sequence at the
  j=3 stripe of the 4×4 unroll, not the store itself.

**Probe P1: pyref kv_mem & delta finite-check** (host-side, no XPU):
on the layer-0 worst capture, conservative bounds:
  - state_pre max abs = 14.70
  - qkvz max abs = 17.50
  - conv_w max abs = 0.406 (width=4) → |k_post_conv| ≤ 28.4
  - |kv_mem| upper bound = max|state| × max|k| × head_k_dim
    = 14.70 × 28.4 × 128 = **5.35e4** (33 orders of magnitude below
    fp32 overflow)
  - beta = sigmoid(-100) = 3.72e-44 (subnormal; flushed to 0 with FTZ
    or treated as the literal subnormal otherwise — both produce
    delta < 1.6e-42, far below any normal threshold)
PASS: delta=0 at every j, including j=3, is robust in IEEE math.
The codegen-bug framing stands; no kv_mem overflow / NaN-via-flushing
escape hatch.

**Probe P2: IS_SPEC=false load-readback** (NATIVE_LAUNCHER,
multi-token, same probe as T19, dispatcher routed via no
spec_state_indices/num_accepted_tokens):
  - Setup: 4-token spans via spec qsl reused as non_spec_qsl;
    one slot per batch (= spec ring's last accepted slot).
    g=1, beta=0 same as T19.
  - Result: post[load_slot] vs pre[load_slot] **byte-equal** across
    all batches. max=0.0e0, hot=0, top-v all zero.

This is a major reframe vs T23/T24's "RA differs, scheduler differs,
can't localize". The IS_SPEC=false template runs the SAME source
line 241 and produces correct machine code; the IS_SPEC=true
template produces buggy machine code on the same source. Same
kernel, different template instantiation, different RA, different
ISA, different bug.

**Refined picture**:
- The bug is NOT in source-level math (P1 confirms delta=0 robustly).
- The bug is NOT in line 241's source pattern per se (P2 confirms
  IS_SPEC=false compiles it correctly).
- The bug IS in IGC's lowering of line 241 *under the IS_SPEC=true
  RA*, which carries two extra captured-by-value parameters
  (spec_state_indices, num_accepted_tokens) that bump live-range
  pressure (per T23's ASM).

**Tick 26 plan: volatile-temp ASM diff** (the user's preferred
alternative to self-assign — incremental rebuild ~270s).

The reviewer-rejected T25 self-assign is no good because IGC elides
the no-op store entirely (T22 drift=0), triggering RA-wide reflow.
The volatile-temp variant emits the store but with a runtime value
IGC can't fold:
```cpp
volatile float prod = k_local[i] * delta[j];
state_local[j * k_bucket_size + i] += prod;
```
T21 already showed this drifts the SAME amount as pristine (9.22),
so RA is effectively unchanged. Diffing pristine spec=true ASM vs
volatile-temp spec=true ASM isolates ONLY the line-241 codegen
perturbation, not the RA-wide reflow that wrecked the T24 diff.

Steps:
1. Edit `csrc/xpu/gdn_attn/gated_delta_rule.hpp:241` to the
   volatile-temp variant above.
2. `nix develop ~/Projects/vllm-xpu-kernels -c vllm-xpu-rebuild`
   (~270s incremental).
3. Re-run `vllm/.spec-gdn-load-readback.py` first to confirm drift
   is still 9.22 (sanity: same bug, RA largely unchanged).
4. Extract ASM via T23's pipeline:
   - `objcopy --dump-section OFFLOAD_DEVICE_CODE=raw _xpu_C.abi3.so`
   - walk archives → ar11 → `64.bmg` member
   - parse ELF section names; pick `.text._ZTSN3gdn23gated_delta_rule_kernelI...Lb1EEE`
     for bf16/bf16 k=4 IS_SPEC=true (the same section used in T23)
   - `iga64 -d -p=2 <bin> -o volatile_spec_true.asm`
5. `diff -u .spec_gdn_disasm/spec_true_k4_bf16.asm volatile_spec_true.asm`
   — register-normalize first if RA shifts; the localized diff
   should be a 4-iteration `mad`/`mul`+`mov` cluster at the j=3
   stripe. Patterns to watch for: GRF-bank conflict (registers in
   the same bank back-to-back), SWSB-token reuse without barrier,
   FMA-then-mov-from-same-grf (write-after-read latency).
6. Revert source post-experiment.

**Tick 27 fallback (if T26 ASM diff is inconclusive)**: JIT vs AOT.
`VLLM_XPU_AOT_DEVICES="" VLLM_XPU_XE2_AOT_DEVICES="" vllm-xpu-build`
(full rebuild ~1h45m, no AOT in offload bundle so runtime falls
back to JIT'd SPIR-V). Re-run probe. If drift=0 under JIT, bug is
purely IGC-AOT; ship JIT-only for this kernel as workaround AND
file Intel bug with concrete ISA evidence.

**Tick 28+ fallback options (workaround ladder)**:
- B. `[[intel::reqd_sub_group_size(16)]]` on the kernel — halves
  SIMD width, re-runs RA. If clean, ship with documented perf cost.
- C. Source-level RA decouple: pass spec_state_indices /
  num_accepted_tokens via a struct or restructure so IS_SPEC=true
  RA pressure matches IS_SPEC=false.
- D. IGC env workarounds: `IGC_DisableLoopUnroll`,
  `IGC_DisableSendS_LSC`, `IGC_ForceOCLSimdWidth=16`, etc.
  Runtime-only, no rebuild.

Tools added: `vllm/.spec-gdn-tick25-cheapprobes.py` (combined
P1+P2; reusable).

Source state: pristine. .so is fresh from background "Incremental
rebuild with pristine source" task that completed during this tick.


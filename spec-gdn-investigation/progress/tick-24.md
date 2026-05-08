2026-05-07 tick 24: inner-loop region located; drift is BOTH
register-allocation AND instruction-scheduling. Cannot localize the
bad ISA pattern from spec=true vs spec=false alone — need a
"good spec=true" build to diff against.

**Inner-loop bounds** (per-token loop body, fully unrolled):
- spec_true_k4_bf16.asm: lines 490 (L6656) → 811 (L10408+`jmpi` at
  1038 to L4952, the per-token-loop top). 322 ASM lines.
- spec_false_k4_bf16.asm: lines 513 (L6920) → 839 (L10776). 327
  lines.

**Vectorization observation**: kernel is compiled SIMD-32 with
M0/M16 dual-issue. Inner loop is NOT 16+16 independent SIMD-16
fmas. Compiler reordered as:
1. State updates: SIMD-16 `mul rN:f bf16*scalar_f` with broadcast
   for `delta[j]` — clusters around lines 619-680 in spec_true.
2. Res updates: 4 dot products via `acc0/acc1` chaining
   (`mul→mad→mad→mad`-final-write) — clusters at lines 555-604 in
   spec_true. Each dot product reduces 4 i-values into one
   `res[j]`. M0+M16 dual-issue means 2 dot products per cluster.

So the in-asm order is: (some res-prep via acc) → state updates →
res-finalize. The compiler interleaved them, contradicting "first
all state, then all res".

**Diff after register normalization** (`r\d+\.\d+` → `rN.M`,
`acc\d+\.\d+` → `accN.M`, `L\d+` → `LN`): still differs.
Instruction *order* differs at the inner-loop entry block. Example
at L6656/L6920 entry:
- spec_false: `mul:d`, `macl:d`, `mul:f -:f`, `add:d`,
  `mul:f * 1.442695e+00:f` (= log2(e), gating exp), `mul:d`,
  `macl:d`, `rndz:f` … then SIMD-32 `add:d` ptr arithmetic.
- spec_true: `add:d`, `or:d 1:w`, `mul:d`, `macl:d`, `or:d 2:w`,
  `add:d`, `or:d 3:w`, `add:d`×2, SIMD-16 `mov:ud`×2, `add:d`,
  `cmp(lt)f2.0:f -105.0:f`, `cmp(gt)f1.0:f 105.0:f`, `mov:ud`×2,
  `shl:q 1:w`, `mov:ud`, … (clamp-and-shift first, exp-prep
  later).

The `cmp` against ±105.0 (exp-input clamping for `exp(A_log*dt)`
to avoid `expf` overflow at f32 ±88-ish, or denorm at < -88) is
present early in spec_true but pushed later in spec_false.
Scheduling drift on top of register drift.

**Conclusion**: cannot localize the bad ISA pattern from
spec=true vs spec=false — they're TWO independent variables (RA +
scheduler) and the difference is correlated, not causal. Need a
controlled experiment: rebuild with a known-good source variant
(`state_local[idx] = state_local[idx]` self-assign at line 241,
which had drift=0 per tick 22's table) and diff that ISA against
current spec=true. Same kernel template, only line 241 changes →
the ISA delta localizes which instruction(s) IGC mis-lowers.

**Next tick (25)** — controlled experiment build (~270s
incremental rebuild):
1. Edit `csrc/xpu/gdn_attn/gated_delta_rule.hpp:241` to
   `state_local[j * k_bucket_size + i] = state_local[j * k_bucket_size + i];`
2. `nix develop ~/Projects/vllm-xpu-kernels -c vllm-xpu-rebuild`
3. Re-extract `_xpu_C.abi3.so` → ar11 → bf16/bf16 k=4 spec=true
   `.text` section → `iga64 -d -p=2`
4. `diff -u current_broken_spec_true.asm new_self_assign_spec_true.asm`
   — register allocation is the SAME (one-line source change),
   so any non-trivial diff is the BUGGY mad/store sequence at
   line 241. Then revert source.

Cache: incremental rebuild keeps existing AOT cache; ~270s.

Source state: pristine (NOT yet edited).


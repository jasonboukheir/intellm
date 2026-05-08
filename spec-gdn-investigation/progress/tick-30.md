2026-05-08 tick 30: T28 IGC env-var ladder — **all four
attempts failed**. Drift bit-equal to baseline JIT every time.

**Setup**: against the JIT-only `_xpu_C.abi3.so` from tick 29
(mtime 1778227678). Each probe used `vllm-run -e <ENV>=<VAL>`
+ `vllm/.spec-gdn-load-readback.py`. NEO compiler cache cleared
between attempts (`rm -rf /root/.cache/neo_compiler_cache/*`)
and `NEO_CACHE_PERSISTENT=0` set; cache hit ruled out.

**Results**:
| Env var(s)                     | max\|delta\| | cells |
|--------------------------------|---------------|-------|
| baseline JIT (no env)          | 8.9465        | 9011  |
| `IGC_ForceOCLSimdWidth=16`     | 8.9465        | 9011  |
| `IGC_DisableLoopUnroll=1`      | 8.9465        | 9011  |
| `IGC_OptDisable=1`             | 8.9465        | 9011  |
| `IGC_ShaderDumpEnable=1`       | 8.9465        | 9011  |

All five runs are *bit-identical* in cell count, max delta, and
top-10 (h, v) profile. Same v=79 hot lane, h=14/19/18/15
dominant.

**Why each failed**:
1. `IGC_ForceOCLSimdWidth=16` — silently overridden by
   `[[sycl::reqd_sub_group_size(sub_group_size)]]` at
   `gated_delta_rule.hpp:87` (`sub_group_size = 32` constexpr).
   The kernel attribute pins SIMD-32, env can't widen it.
   Confirmed via shader dump: `OCL_asm*_simd32_entry_0003.asm`
   was emitted (not simd16).
2. `IGC_DisableLoopUnroll=1` — IGC's auto-unroll only. The 4×4
   inner loop is hard-coded `#pragma unroll` (lines 232/236/238
   in `gated_delta_rule.hpp`); IGC must honor the pragma so this
   env has no path to disable the unroll that contains line 241.
3. `IGC_OptDisable=1` — drift still bit-identical to -O3. The
   bug survives even under disabled optimization. Either: (a) the
   bug is below all IGC optimization passes, in the SPIR-V →
   GEN ISA lowering itself, or (b) `IGC_OptDisable` only gates
   mid-IR opts and the codegen-level scheduler/regalloc that
   produces the bug runs unconditionally. Either is consistent
   with T26's observation that mad-count was preserved (52/52)
   but RA differed.
4. `IGC_ShaderDumpEnable=1` — dump verification only; produced
   746 files in `/tmp/igcdump/` (inside the container). Confirms
   IGC IS reading env vars in this build/runtime configuration,
   so failures of (1)-(3) are not "env vars ignored" — they're
   "the bug is at a level these env vars don't reach".

**Conclusion**: env-var workaround track is exhausted. Cannot
fix from the IGC command line on this build. Must change source.

**T29 next**: source-level RA decouple. Two parallel paths to
try (independently or in combo):

A. **Force SIMD-16 by source change**: edit
   `gated_delta_rule.hpp:7` `sub_group_size = 32` → `16`.
   This will require:
   - k_bucket_size becomes 8 instead of 4 for head_k_dim=128
     (k_bucket_size = head_k_dim / sub_group_size).
   - The k=8 IS_SPEC=true template instantiation already exists
     (`_ZTSN3gdn23gated_delta_rule_kernelIN4sycl3_V13ext6oneapi8bfloat16ES5_Li8ELb1EEE`
     27968 bytes in ar11; see tick 28 enumeration) — code path
     compiles.
   - Caller dispatch in `gdn_attn_interface.cpp` may need an
     update if k=4 was hardcoded; verify.
   - Expected ~50% perf cost (SIMD-16 halves throughput).
   - Incremental rebuild (~270s, vllm-xpu-rebuild — but use the
     wrapper-bypass `docker exec` route per tick 27, since the
     wrapper rewrites empty AOT vars).

B. **Struct-wrap spec params** (T22 RA hypothesis): combine
   `spec_state_indices_tensor` + `num_accepted_tokens` into one
   struct passed by value. T23's evidence was that the
   IS_SPEC=true template carries two extra captured-by-value
   parameters that bump live-range pressure vs IS_SPEC=false.
   Consolidating might equalize the pressure and avoid the
   buggy RA path.
   - Probably the smaller perf hit if it works.
   - Riskier: adds a new layout, needs careful interface changes
     in the dispatcher.

**Plan for tick 31**: try A first (simpler change, more confident
that SIMD-16 changes RA enough to bypass the bug). If A fixes
drift, rung 4 turns GREEN with documented perf cost. If A
doesn't fix, try B.

Source state: pristine. .so: JIT-only pristine.

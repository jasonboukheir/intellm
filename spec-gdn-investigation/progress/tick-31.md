2026-05-08 tick 31: **T29.A — SIMD-16 source change FAILED**.
T22's RA-pressure hypothesis is falsified.

**Setup**: edited `gated_delta_rule.hpp:7`
`sub_group_size = 32` → `16`. Incremental rebuild
(`docker exec ... -e FA2_KERNELS_ENABLED=OFF MAX_JOBS=2 pip
install -e .`, ~12 min, JIT-only). `_xpu_C.abi3.so` mtime
8480 → 8480 (rebuilt). NEO cache cleared between probe runs.

**Verification of source effect**:
- IGC shader dumps under `IGC_ShaderDumpEnable=1`: 100 distinct
  kernels, all `_simd16_entry_*`. Specifically:
  `OCL_asmef7e1cb966451be5_simd16_entry_*.visaasm` includes
  `gated_delta_rule_kernel<bfloat16, bfloat16, 8, true>`
  (mangled `_ZTSN3gdn23gated_delta_rule_kernelIN4sycl3_V13ext6oneapi8bfloat16ES5_Li8ELb1EEE`).
  This is the bf16/bf16 k=8 IS_SPEC=true template — *different
  template instantiation* than the original buggy k=4
  (head_k_dim=128 / sub_group_size=16 = k_bucket_size=8).
- `simd32_entry` dumps that also appeared belong to
  `chunk_gated_delta_rule_xe2` (the prefill kernel — different
  source file `xe_2/chunk_causal_conv1d_xe2.hpp:14`, sub_group
  unchanged) and other unrelated XE2 kernels.
- So the SIMD-16 change DID take effect for the kernel under
  test.

**Probe result**: `vllm/.spec-gdn-load-readback.py`
  - max |delta| = **8.9465**
  - 9011 cells, v=79(31), h=14(116) / h=19(115) / h=18(112)…
  - **Bit-identical to T27 JIT pristine, T26 AOT volatile-temp,
    and all four T28 IGC env vars.**

**Implication — T22 falsified**: T22 hypothesized "IS_SPEC=true
RA pressure (extra captured-by-value `spec_state_indices` +
`num_accepted_tokens`) tips IGC into a buggy register
allocation". But:
  - SIMD-16 halves subgroup width, dramatically restructures RA
    (different live ranges, different physical registers,
    different scheduling).
  - SIMD-16 also dispatches k=8 instead of k=4 — *different
    template instantiation*, different kernel binary, different
    inner-loop unroll factor (`v_dim_per_sg=4 × k_bucket_size=8
    = 32` instead of 16).
  - Under all of that, the bug expresses with the *exact same
    drift number* (8.9465) and *exact same statistical
    fingerprint* (head/v profile bit-identical).

The bug is NOT RA-pressure-related, NOT template-instantiation-
specific (k=4 vs k=8), NOT SIMD-width-specific. It survives
every codegen-level perturbation tried.

**Suspicious invariance**: across T26 (AOT volatile-temp), T27
(JIT pristine), T28 (4 IGC env vars), T29.A (SIMD-16 k=8), the
drift is *bit-equal* — same 9011 cells, same 8.9465 max, same
top-10 head/v profile. Only T19 (the original AOT pristine
build) measured 9.22; *every* rebuild since then lands at 8.95.
Either:
  - 9.22 was a transient artifact of the T19-era build (different
    compiler version, different cmake state, different cache),
    and 8.9465 is the canonical bug fingerprint.
  - The probe is reaching a deterministic codepath that the
    rebuild process locks in identically every time.

**Next direction (tick 32) — methodological sanity check**:
Before continuing the workaround ladder, validate that we're
actually testing what we think we are. Make a *deliberately
breaking* source change to line 241 and confirm the drift
changes:

```cpp
// Replace state += k*delta with state = 0
state_local[j * k_bucket_size + i] = 0.0f;  // not the original
```

Expected: drift jumps to ~max-magnitude of captured state
(~14.7) because state_local is zeroed before writeback. If
drift instead stays at 8.9465, our build/probe pipeline isn't
actually exercising the kernel we think it is — and every
prior conclusion is suspect.

This is cheap: incremental rebuild ~5-10 min, probe ~30 sec.
Critical step before further workaround attempts.

If sanity-check passes (drift changes as expected): try T29.B
(struct-wrap spec params) — much weaker test of RA-pressure
than SIMD-16 already was, so prior odds are bad, but worth a
try since other paths exhausted.

If sanity-check fails: re-audit the build/install/load path.
Possibilities to check: editable install symlink resolution,
NEO cache layering, kernel selection in `_call_sycl`,
`_xpu_ops.gdn_attention` C++ binding.

**Source state**: reverted `sub_group_size` to 32. Hpp pristine.
.so still has the SIMD-16 build (will rebuild in tick 32 with
the line-241 sanity-check change).

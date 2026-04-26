# XMX investigation — joint_matrix dead-end on this Battlemage

## What I tried

Wrote a flash-attention v2 forward kernel using `sycl::ext::oneapi::experimental::matrix::joint_matrix` for the QK^T and PV matmuls (DPAS shape M=8, N=16, K=16, FP16 → FP32). See `src/kernels/flash_attention_xmx.hpp`.

## What happened

- Compiles fine with `icpx -fsycl`.
- AOT-targeted `intel_gpu_bmg_g31` (Battlemage) builds cleanly.
- At kernel launch the runtime throws:
  ```
  what():  no matrix hardware on the target device, joint_matrix is not supported
  ```

## Root cause

The device reports **zero** matrix combinations through the SYCL info query:

```cpp
auto combos = dev.get_info<
    sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
// combos.size() == 0 on Intel(R) Graphics [0xe223], driver 25.48.36300.8
```

(See `scripts/probe_matrix.cpp` to re-run.) Battlemage absolutely has DPAS — the open-source DPC++ `joint_matrix` lowering for Xe2 just isn't wired up in this stack yet.

## What SYCL*TLA does

Skips `joint_matrix` entirely. Goes straight to inline SPIR-V intrinsics via CUTE's `XE_DPAS_TT` atom (`include/cute/arch/mma_xe.hpp`) — emits `dpas.bf.bf.8.<simd>` instructions in inline asm. That's why the BMG flash-attention example (`examples/06_bmg_flash_attention`) hits ~78% peak: it bypasses the portable matrix API.

## Paths forward

1. **Direct SPIR-V intrinsics.** Write our own thin DPAS wrapper using `__builtin_IB_sub_group_idpas_*` builtins. Same approach as CUTE but minimal infrastructure. Most likely the right path.
2. **ESIMD.** Intel's Explicit-SIMD extension exposes XMX directly. Different programming model (vector-style, not SIMT), more invasive rewrite.
3. **Adopt sycl-tla.** Use CUTE's atoms wholesale. Heavyweight integration; we'd be re-implementing pieces of the existing BMG example.
4. **Wait for joint_matrix.** Track Intel/llvm — there are ongoing PRs adding Xe2 DPAS lowering for `joint_matrix`. Not a fix we control.

For the next iteration I'd go with (1): a small `xfa::dpas_8_16_16_f16_f32(C, A, B)` helper using the SPIR-V builtin, called from the existing kernel structure.

## State of the XMX kernel in this repo

`src/kernels/flash_attention_xmx.hpp` is committed but disabled in the test
suite by default. To re-test once joint_matrix gains Xe2 support:

```
XFA_TEST_XMX=1 ./build/tests/test_flash_attn
```

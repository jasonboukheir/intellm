# xpu-kvcache-quant

KV cache quantization for Intel Xe2 GPUs — ports TurboQuant and RotorQuant to SYCL/DPC++.

## Goal

Enable KV cache compression on Intel Arc Pro B70 to extend context length and reduce
memory pressure. Neither TurboQuant nor RotorQuant currently supports Intel XPU.

## Architecture

### TurboQuant (Google, ICLR 2026)
- Walsh-Hadamard Transform for random orthogonal rotation
- Lloyd-Max optimal scalar quantization
- QJL projection for residual sign information (1 extra bit)
- Group quantization with per-group scaling
- Bit-packing for storage density
- Target: 3-bit keys + 2-bit values (~4.4x compression)
- Reference: https://github.com/0xSero/turboquant (GPL-3.0)

### RotorQuant (Scrya)
- Clifford algebra Cl(3,0) rotors instead of Walsh-Hadamard
- ~100 FMAs per vector vs 16,384 for d=128 WHT
- Three strategies: IsoQuant (quaternion 4D), PlanarQuant (Givens 2D), RotorQuant (Clifford 3D)
- 28% faster decode, 5.3x faster prefill vs TurboQuant
- Reference: https://github.com/scrya-com/rotorquant
- vLLM integration requested: https://github.com/vllm-project/vllm/issues/38291

## Source layout

- `src/turboquant/` — TurboQuant SYCL kernels
  - `kernels/walsh_hadamard.hpp` — WHT rotation
  - `kernels/lloyd_max.hpp` — Optimal scalar quantization
  - `kernels/qjl.hpp` — QJL residual projection
  - `kernels/bitpack.hpp` — Bit-packing
- `src/rotorquant/` — RotorQuant SYCL kernels
  - `kernels/clifford_rotor.hpp` — Cl(3,0) sandwich product
  - `kernels/planar_quant.hpp` — Givens rotation (PlanarQuant variant)
  - `kernels/iso_quant.hpp` — Quaternion rotation (IsoQuant variant)
- `src/common/` — Shared quantization utilities
- `src/bindings/` — Python bindings (pybind11)
- `vllm_adapter/` — vLLM monkey-patch integration
- `benchmarks/` — Performance comparison: TurboQuant vs RotorQuant vs no compression

## Hardware target

Intel Arc Pro B70: 32GB GDDR6 @ 608 GB/s. KV cache compression directly translates
to longer context (32GB is generous but finite at FP16 for large models).

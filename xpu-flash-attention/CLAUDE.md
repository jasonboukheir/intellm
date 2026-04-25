# xpu-flash-attention

Flash attention kernel implementations for Intel Xe2 (Battlemage) GPUs using SYCL/DPC++.

## Goal

Optimize flash attention for Intel Arc Pro B70 and contribute improvements upstream
to vllm-xpu-kernels. Current state of art is ~78% of peak on Battlemage via SYCL*TLA.

## Architecture

- `src/kernels/` — SYCL kernel implementations
  - `flash_attention_xe2.hpp` — Xe2-optimized flash attention v2
  - Reference: SYCL*TLA flash attention, vllm-xpu-kernels attention variants
- `src/bindings/` — pybind11 Python bindings for benchmarking/testing
- `benchmarks/` — Performance measurement against baseline vllm-xpu-kernels
- `tests/` — Correctness tests against reference PyTorch implementation

## Key references

- SYCL*TLA: https://github.com/intel/sycl-tla (foundation library)
- vllm-xpu-kernels: https://github.com/vllm-project/vllm-xpu-kernels (upstream target)
- Dao-AILab flash-attention Intel PR: https://github.com/Dao-AILab/flash-attention/pull/1528

## Build

SYCL kernels require Intel DPC++ compiler (icpx -fsycl). Enter oneAPI env first:
```
nix develop .#oneapi   # or: nix run ../nix-intel-xpu#oneapi-env
source /opt/intel/oneapi/setvars.sh
cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Hardware target

Intel Arc Pro B70: 32 Xe-cores, 256 XMX engines, 32GB GDDR6, 608 GB/s, Xe2 arch.

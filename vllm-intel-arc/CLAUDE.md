# vllm-intel-arc

Integration project for running vLLM on Intel Arc Pro B70 with custom kernels.

## Goal

Provide the orchestration layer for:
1. Running Intel's official vLLM container on Arc Pro B70
2. Integrating xpu-flash-attention and xpu-kvcache-quant custom kernels
3. Benchmarking baseline vs custom kernel performance
4. Comparing metrics across model sizes and configurations

## Architecture

- `container/` — Dockerfiles extending intel/vllm with custom kernels
- `configs/models/` — Per-model serving configs (quantization, tensor parallel, etc.)
- `benchmarks/` — Automated benchmark suite
- `scripts/` — Container management, server lifecycle, benchmark orchestration
- `tests/` — Integration tests (model loading, inference correctness)

## Base container

`intel/vllm:0.17.0-xpu` — latest Intel-optimized vLLM for XPU.
See: https://hub.docker.com/r/intel/vllm
Source: https://github.com/intel/ai-containers/tree/main/vllm

## Supported precision on B70

- FP16/BF16: full support
- Dynamic FP8: supported
- MXFP4: supported (best throughput)
- INT4: supported
- AWQ/GPTQ: NOT available on XPU

## Hardware

Intel Arc Pro B70: Xe2, 32 Xe-cores, 32GB GDDR6 @ 608 GB/s, PCIe 5.0 x16.

## Key references

- vLLM XPU docs: https://docs.vllm.ai/en/stable/models/hardware_supported_models/xpu/
- vLLM Intel Arc blog: https://vllm.ai/blog/intel-arc-pro-b
- vllm-xpu-kernels: https://github.com/vllm-project/vllm-xpu-kernels
- Intel AI containers: https://github.com/intel/ai-containers

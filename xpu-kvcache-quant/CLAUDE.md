# xpu-kvcache-quant

SYCL kernels for KV-cache quantization (TurboQuant + RotorQuant family) on Intel XPU. Roadmap and phase breakdown live in `PLAN.md` — read that before starting work.

Current state in one line: rotation kernels mature, Lloyd-Max bit-packing is TODO, QJL is a stub, pybind11 + vllm_adapter are scaffolds. Plan is to wire these behind the merged upstream vLLM TurboQuant backend (PR #38479, 2026-04-15) rather than ship a custom monkey-patch.

## References

- Upstream merged backend: https://github.com/vllm-project/vllm/pull/38479
- TurboQuant standalone (Triton, GPL-3.0): https://github.com/0xSero/turboquant
- TurboQuant pure-PyTorch oracle (MIT): https://github.com/scos-lab/turboquant
- RotorQuant: https://github.com/scrya-com/rotorquant
- RotorQuant integration request in vLLM: https://github.com/vllm-project/vllm/issues/38291

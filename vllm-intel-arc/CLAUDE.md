# vllm-intel-arc

Base container: `intel/vllm:0.17.0-xpu`

AWQ/GPTQ are NOT available on XPU. Supported: FP16/BF16, dynamic FP8, MXFP4, INT4.

## References

- vLLM XPU docs: https://docs.vllm.ai/en/stable/models/hardware_supported_models/xpu/
- vLLM Intel Arc blog: https://vllm.ai/blog/intel-arc-pro-b
- vllm-xpu-kernels: https://github.com/vllm-project/vllm-xpu-kernels
- Intel AI containers: https://github.com/intel/ai-containers

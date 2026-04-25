ARG BASE_IMAGE=intel/vllm:0.17.0-xpu

FROM ${BASE_IMAGE}

# Custom kernel integration layer
# Extends Intel's official vLLM container with our flash attention
# and KV cache quantization kernels.

# Install custom flash attention
# TODO: Once xpu-flash-attention is compiled, copy the shared library
# COPY --from=flash-attn-builder /build/lib/xpu_flash_attn.so /usr/local/lib/python3.*/dist-packages/

# Install custom KV cache quantization
# TODO: Once xpu-kvcache-quant is compiled, copy the shared library + Python adapter
# COPY --from=kvcache-quant-builder /build/lib/xpu_kvcache_quant.so /usr/local/lib/python3.*/dist-packages/
# COPY vllm_adapter/ /usr/local/lib/python3.*/dist-packages/xpu_kvcache_quant/vllm_adapter/

# Install Python utilities for benchmarking
COPY benchmarks/ /workspace/benchmarks/
COPY configs/ /workspace/configs/

WORKDIR /workspace

# Verify XPU is accessible
RUN python3 -c "import torch; import intel_extension_for_pytorch; print(f'XPU devices: {torch.xpu.device_count()}')" 2>/dev/null || echo "No XPU device available during build (expected)"

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]

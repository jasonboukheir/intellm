"""vLLM adapter for XPU KV cache quantization.

Monkey-patches vLLM's attention backend to use TurboQuant or RotorQuant
KV cache compression on Intel XPU.

Usage:
    # Before starting vLLM server:
    import xpu_kvcache_quant.vllm_adapter
    xpu_kvcache_quant.vllm_adapter.patch(method="rotorquant", key_bits=3, value_bits=2)

    # Then start vLLM normally
    # The KV cache will be compressed transparently during inference.
"""

from .monkey_patch import patch, unpatch

__all__ = ["patch", "unpatch"]

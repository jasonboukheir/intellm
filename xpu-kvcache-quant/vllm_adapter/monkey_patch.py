"""Monkey-patch vLLM to use quantized KV cache on Intel XPU.

This follows the same pattern as TurboQuant's vLLM integration:
intercept the KV cache write/read paths to compress/decompress on the fly.

Target integration points in vLLM:
  - vllm.attention.backends.xpu.XPUAttentionBackend
  - vllm.worker.xpu_worker.XPUWorker (cache allocation)

The patch needs to:
  1. Reduce KV cache allocation size (compressed storage)
  2. Intercept cache_k/cache_v writes to quantize before storing
  3. Intercept cache_k/cache_v reads to dequantize before attention
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

_original_fns = {}
_patched = False


@dataclass
class KVQuantConfig:
    method: str = "rotorquant"  # turboquant, isoquant, planarquant, rotorquant
    key_bits: int = 3
    value_bits: int = 2
    group_size: int = 128
    use_qjl: bool = True  # TurboQuant only

    @property
    def key_compression_ratio(self):
        return 16.0 / self.key_bits  # from FP16

    @property
    def value_compression_ratio(self):
        return 16.0 / self.value_bits

    @property
    def overall_compression_ratio(self):
        return 2.0 / (1.0 / self.key_compression_ratio + 1.0 / self.value_compression_ratio)


_config: Optional[KVQuantConfig] = None


def patch(method="rotorquant", key_bits=3, value_bits=2, **kwargs):
    """Patch vLLM to use quantized KV cache.

    Call this before initializing vLLM engine.

    Args:
        method: One of "turboquant", "isoquant", "planarquant", "rotorquant"
        key_bits: Quantization bits for keys (2-4, default 3)
        value_bits: Quantization bits for values (2-4, default 2)
    """
    global _patched, _config

    if _patched:
        logger.warning("KV cache quantization already patched")
        return

    _config = KVQuantConfig(method=method, key_bits=key_bits, value_bits=value_bits, **kwargs)

    logger.info(
        f"Patching vLLM KV cache: method={method}, "
        f"K={key_bits}bit, V={value_bits}bit, "
        f"compression={_config.overall_compression_ratio:.1f}x"
    )

    try:
        _patch_xpu_attention()
        _patched = True
        logger.info("KV cache quantization patch applied successfully")
    except ImportError as e:
        logger.error(f"Failed to patch vLLM — is vLLM installed? {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to patch vLLM: {e}")
        unpatch()
        raise


def unpatch():
    global _patched, _config
    for name, fn in _original_fns.items():
        _restore_fn(name, fn)
    _original_fns.clear()
    _patched = False
    _config = None
    logger.info("KV cache quantization patch removed")


def _patch_xpu_attention():
    """Patch the XPU attention backend's KV cache operations."""
    # TODO: Implement once the SYCL kernels are compiled and bindings work
    #
    # The integration points are:
    #
    # 1. Cache allocation (reduce memory):
    #    vllm.worker.xpu_worker.XPUCacheEngine.allocate_gpu_cache
    #    -> Allocate compressed-size tensors instead of full FP16
    #
    # 2. Cache write (quantize):
    #    vllm.attention.backends.xpu.XPUAttentionBackend.forward
    #    -> After computing K,V, quantize before storing in cache
    #    -> key_cache = rotate(K) -> quantize(rotated_K)
    #    -> val_cache = rotate(V) -> quantize(rotated_V)
    #
    # 3. Cache read (dequantize):
    #    vllm.attention.backends.xpu.XPUAttentionBackend.forward
    #    -> Before attention computation, dequantize from cache
    #    -> K = dequantize(key_cache) -> inverse_rotate(K)
    #
    # For now, this is a stub that logs the intended patches.

    logger.warning(
        "KV cache quantization kernels not yet compiled — "
        "running in dry-run mode (no actual compression)"
    )


def _restore_fn(dotted_name, fn):
    parts = dotted_name.rsplit(".", 1)
    if len(parts) == 2:
        import importlib
        mod = importlib.import_module(parts[0])
        setattr(mod, parts[1], fn)

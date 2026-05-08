"""Triton workarounds for the Intel XPU backend that are needed to run the
FLA prefill kernel on Qwen3.6 shapes (the spec-GDN replay oracle).

Two issues, two fixes:

1. The `TritonIntelStrideVersioning` pass crashes the MLIR pipeline with
   `PassManager::run failed` on `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`
   for every autotune BV/num_warps/num_stages config we tried. The pass is
   added unconditionally in `triton/backends/intel/compiler.py:make_ttir`
   with no env knob. Stride versioning is an *optimization* (specializes
   code on stride values for better codegen) — disabling it costs perf
   but is correctness-safe. We monkey-patch
   `intel.passes.ttir.add_stride_versioning` to a no-op.

2. Triton's `Autotuner._bench` only catches OutOfResources /
   CompileTimeAssertionFailure / PTXASError; the generic RuntimeError from
   the MLIR failure bypasses that filter and kills EngineCore. We broaden
   the catch so a failing config records `inf` time and the autotuner picks
   a different one. (Belt-and-braces: kept even with #1 in place because
   other Intel passes raise the same exception class.)

Loaded via a `.pth` file so the patch runs at Python site init, before
vllm/triton imports happen. Source-of-truth lives in the bind-mounted
vllm tree; staged into the container at
/opt/venv/lib/python3.12/site-packages/_spec_gdn_triton_patch.{py,pth}
(lost on `vllm-xpu-clean`; restage from this file).
"""

import functools


def _disable_stride_versioning() -> None:
    """No-op the Intel TTIR stride-versioning pass."""
    try:
        from triton._C.libtriton import intel as _intel  # type: ignore
    except Exception:
        return
    try:
        ttir = _intel.passes.ttir
    except Exception:
        return
    if getattr(ttir, "_stride_versioning_disabled", False):
        return
    if hasattr(ttir, "add_stride_versioning"):
        def _noop(_pm):
            return None
        ttir.add_stride_versioning = _noop
        ttir._stride_versioning_disabled = True


def _broaden_autotuner_catch() -> None:
    """Make Autotuner._bench survive RuntimeError-flavored compile failures."""
    try:
        from triton.runtime import autotuner as _at
    except Exception:
        return

    if getattr(_at.Autotuner, "_bench_runtime_error_safe", False):
        return

    original_bench = _at.Autotuner._bench

    @functools.wraps(original_bench)
    def patched_bench(self, *args, config=None, **kwargs):
        try:
            return original_bench(self, *args, config=config, **kwargs)
        except RuntimeError as e:
            msg = str(e)
            if (
                "PassManager::run failed" in msg
                or "Failures have been detected while processing an MLIR pass" in msg
                or "TritonIntelStrideVersioning" in msg
            ):
                return [float("inf"), float("inf"), float("inf")]
            raise

    _at.Autotuner._bench = patched_bench
    _at.Autotuner._bench_runtime_error_safe = True


_disable_stride_versioning()
_broaden_autotuner_catch()

"""Tick-45 rung 11 perf probe: SYCL vs FLA throughput comparison.

Time the SYCL spec gdn_attention path and the FLA inline oracle on the
same capture. Run each N times, report median latency and tok/s.

Rung 11 success: SYCL spec tok/s >= FLA spec tok/s.
"""

from __future__ import annotations

import os
import sys
import time

import torch

import vllm._xpu_ops  # noqa: F401
import vllm_xpu_kernels._xpu_C  # noqa: F401

sys.path.insert(0, "/workspace/vllm")

from tests.kernels.xpu.test_spec_gdn_replay import (  # noqa: E402
    _build_dense_pool,
    _call_sycl,
    _compute_fla_spec_oracle,
)

CAPTURE = os.environ.get(
    "SPEC_GDN_CAPTURE",
    "/tmp/spec_gdn_captures/"
    "tuple_language_model_model_layers_0_linear_attn_000004_spec_K4_min4_max4.pt",
)
N_WARMUP = int(os.environ.get("SPEC_GDN_PERF_WARMUP", "3"))
N_RUNS = int(os.environ.get("SPEC_GDN_PERF_RUNS", "20"))


def _xpu_sync():
    if hasattr(torch.xpu, "synchronize"):
        torch.xpu.synchronize()


def time_sycl(payload, device):
    times = []
    for _ in range(N_WARMUP):
        conv_pool, ssm_pool, remap, _ = _build_dense_pool(payload, device)
        _call_sycl(payload, conv_pool, ssm_pool, remap, device)
        _xpu_sync()
    for _ in range(N_RUNS):
        conv_pool, ssm_pool, remap, _ = _build_dense_pool(payload, device)
        _xpu_sync()
        t0 = time.perf_counter()
        _call_sycl(payload, conv_pool, ssm_pool, remap, device)
        _xpu_sync()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return sorted(times)


def time_fla(payload, device):
    times = []
    for _ in range(N_WARMUP):
        _compute_fla_spec_oracle(payload, device)
        _xpu_sync()
    for _ in range(N_RUNS):
        _xpu_sync()
        t0 = time.perf_counter()
        _compute_fla_spec_oracle(payload, device)
        _xpu_sync()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return sorted(times)


def main():
    os.environ.setdefault("VLLM_XPU_USE_SYCL_SPEC_GDN", "1")
    payload = torch.load(CAPTURE, map_location="cpu", weights_only=False)
    payload["__path__"] = CAPTURE
    n_actual = int(payload["num_actual_tokens"])
    device = torch.device("xpu")

    print(f"capture: {os.path.basename(CAPTURE)}")
    print(f"n_actual tokens: {n_actual}")
    print(f"warmup: {N_WARMUP}, timed runs: {N_RUNS}")
    print()

    sycl_times = time_sycl(payload, device)
    fla_times = time_fla(payload, device)

    sycl_median = sycl_times[N_RUNS // 2]
    sycl_min = sycl_times[0]
    sycl_max = sycl_times[-1]
    fla_median = fla_times[N_RUNS // 2]
    fla_min = fla_times[0]
    fla_max = fla_times[-1]

    sycl_tps = n_actual / sycl_median
    fla_tps = n_actual / fla_median

    print(f"SYCL  median {sycl_median*1e3:.3f}ms  "
          f"min {sycl_min*1e3:.3f}  max {sycl_max*1e3:.3f}  "
          f"tok/s = {sycl_tps:.1f}")
    print(f"FLA   median {fla_median*1e3:.3f}ms  "
          f"min {fla_min*1e3:.3f}  max {fla_max*1e3:.3f}  "
          f"tok/s = {fla_tps:.1f}")
    print()
    ratio = sycl_tps / fla_tps
    print(f"speedup (SYCL / FLA) = {ratio:.2f}x")
    print(f"rung 11 gate (SYCL >= FLA): {'PASS' if ratio >= 1.0 else 'FAIL'}")


if __name__ == "__main__":
    main()

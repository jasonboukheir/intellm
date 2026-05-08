"""Tick-43 determinism probe (rung 9).

Run the SYCL spec gdn_attention path on a single capture 10 times,
confirm byte-equal core_attn_out / conv_state / ssm_state on every
iteration. Each call uses a fresh dense pool so prior iterations don't
contaminate the next.
"""

from __future__ import annotations

import os
import sys

import torch

import vllm._xpu_ops  # noqa: F401
import vllm_xpu_kernels._xpu_C  # noqa: F401

sys.path.insert(0, "/workspace/vllm")

from tests.kernels.xpu.test_spec_gdn_replay import (  # noqa: E402
    _build_dense_pool,
    _call_sycl,
)

CAPTURE = os.environ.get(
    "SPEC_GDN_CAPTURE",
    "/tmp/spec_gdn_captures/"
    "tuple_language_model_model_layers_0_linear_attn_000004_spec_K4_min4_max4.pt",
)
N_RUNS = int(os.environ.get("SPEC_GDN_DETERMINISM_RUNS", "10"))


def main():
    os.environ.setdefault("VLLM_XPU_USE_SYCL_SPEC_GDN", "1")
    payload = torch.load(CAPTURE, map_location="cpu", weights_only=False)
    payload["__path__"] = CAPTURE
    n_actual = int(payload["num_actual_tokens"])
    device = torch.device("xpu")

    runs = []
    for run_id in range(N_RUNS):
        conv_pool, ssm_pool, remap, _ = _build_dense_pool(payload, device)
        core, _, conv_post, ssm_post = _call_sycl(
            payload, conv_pool, ssm_pool, remap, device
        )
        runs.append(
            (
                core[:n_actual].detach().to("cpu").clone(),
                conv_post.detach().to("cpu").clone(),
                ssm_post.detach().to("cpu").clone(),
            )
        )

    print(f"capture: {os.path.basename(CAPTURE)}")
    print(f"runs: {N_RUNS}")
    base_core, base_conv, base_ssm = runs[0]
    all_byte_equal = True
    for run_id in range(1, N_RUNS):
        core, conv, ssm = runs[run_id]
        d_core = (core.float() - base_core.float()).abs().max().item()
        d_conv = (conv.float() - base_conv.float()).abs().max().item()
        d_ssm = (ssm.float() - base_ssm.float()).abs().max().item()
        ok = (d_core == 0.0) and (d_conv == 0.0) and (d_ssm == 0.0)
        if not ok:
            all_byte_equal = False
        print(
            f"  run {run_id} vs run 0: core={d_core:.4e} "
            f"conv={d_conv:.4e} ssm={d_ssm:.4e} "
            f"{'OK' if ok else 'DIVERGED'}"
        )

    if all_byte_equal:
        print("\nDETERMINISM: PASS — every run byte-equal to run 0.")
    else:
        print("\nDETERMINISM: FAIL — at least one run diverged.")


if __name__ == "__main__":
    main()

"""Tick-44 rung 6 probe: synthetic num_accepted_tokens=0 override.

Take a min1_max1 capture, override num_accepted_tokens to 0 (= no spec
tokens accepted from the previous round), run both the SYCL kernel and
the FLA oracle, diff core_attn_out / conv_state / ssm_state.

Exercises the defensive `spec_skip_init_load` path in
causal_conv1d.hpp (lines 165-172) and gated_delta_rule.hpp's
n_acc<=0 branch.
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
    _compute_fla_spec_oracle,
    _remap_index_tensor,
)

CAPTURE = os.environ.get(
    "SPEC_GDN_CAPTURE",
    "/tmp/spec_gdn_captures/"
    "tuple_language_model_model_layers_0_linear_attn_000001_spec_K4_min1_max1.pt",
)


def main():
    os.environ.setdefault("VLLM_XPU_USE_SYCL_SPEC_GDN", "1")
    payload = torch.load(CAPTURE, map_location="cpu", weights_only=False)
    payload["__path__"] = CAPTURE
    n_actual = int(payload["num_actual_tokens"])
    device = torch.device("xpu")

    # Override num_accepted_tokens to 0 (rung 6 case).
    orig_n_acc = payload["num_accepted_tokens"].clone()
    payload["num_accepted_tokens"] = torch.zeros_like(orig_n_acc)
    print(f"capture: {os.path.basename(CAPTURE)}")
    print(f"original num_accepted: {orig_n_acc.tolist()}")
    print(f"forced  num_accepted: {payload['num_accepted_tokens'].tolist()}")

    conv_pool, ssm_pool, remap, _ = _build_dense_pool(payload, device)
    sycl_core, _, sycl_conv_post, sycl_ssm_post = _call_sycl(
        payload, conv_pool, ssm_pool, remap, device
    )
    sycl_core = sycl_core[:n_actual].detach().to("cpu").float()
    sycl_conv_post = sycl_conv_post.detach().to("cpu").float().clone()
    sycl_ssm_post = sycl_ssm_post.detach().to("cpu").float().clone()

    oracle_core, oracle_conv_post, oracle_ssm_post, _ = _compute_fla_spec_oracle(
        payload, device
    )
    oracle_core = oracle_core[:n_actual].detach().to("cpu").float()
    oracle_conv_post = oracle_conv_post.detach().to("cpu").float().clone()
    oracle_ssm_post = oracle_ssm_post.detach().to("cpu").float().clone()

    diff_core = (sycl_core - oracle_core).abs()
    print(
        f"\ncore_attn_out: max={diff_core.max().item():.4e} "
        f"mean={diff_core.mean().item():.4e}"
    )

    spec_idx = payload["spec_state_indices_tensor"]
    spec_idx_dense = _remap_index_tensor(spec_idx, remap).to(torch.long)
    n_seq, K = spec_idx_dense.shape

    print("\n--- per-spec-token conv_state max abs diff ---")
    for batch in range(n_seq):
        for t in range(K):
            slot = int(spec_idx_dense[batch, t].item())
            if slot <= 0 or slot >= sycl_conv_post.shape[0]:
                continue
            d = (sycl_conv_post[slot] - oracle_conv_post[slot]).abs()
            print(
                f"  batch={batch} t={t} slot={slot}: "
                f"max={d.max().item():.4e}, mean={d.mean().item():.4e}, "
                f"#cells>2e-2: {(d > 2e-2).sum().item()}"
            )

    print("\n--- per-spec-token ssm_state max abs diff ---")
    for batch in range(n_seq):
        for t in range(K):
            slot = int(spec_idx_dense[batch, t].item())
            if slot <= 0 or slot >= sycl_ssm_post.shape[0]:
                continue
            d = (sycl_ssm_post[slot] - oracle_ssm_post[slot]).abs()
            print(
                f"  batch={batch} t={t} slot={slot}: "
                f"max={d.max().item():.4e}, mean={d.mean().item():.4e}, "
                f"#cells>2e-2: {(d > 2e-2).sum().item()}"
            )


if __name__ == "__main__":
    main()

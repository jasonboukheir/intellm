"""Tick-36 core_attn_out diff probe.

Run the SYCL spec gdn_attention kernel + the inline FLA oracle on
the layer-0 worst capture, compute element-wise abs diff on
core_attn_out, and report:
  - max abs diff and its (t, h, v) coordinate
  - per-token-axis (t) max diff distribution
  - per-head (h) max diff distribution
  - per-v_dim (v) max diff distribution
  - top-K cell coordinates with their (sycl, fla, diff) values

Goal: localize where in the res-accumulation
(gated_delta_rule.hpp:233-249) SYCL diverges from FLA.
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
)

CAPTURE = os.environ.get(
    "SPEC_GDN_CAPTURE",
    "/tmp/spec_gdn_captures/"
    "tuple_language_model_model_layers_0_linear_attn_000004_spec_K4_min4_max4.pt",
)


def main():
    os.environ.setdefault("VLLM_XPU_USE_SYCL_SPEC_GDN", "1")
    payload = torch.load(CAPTURE, map_location="cpu", weights_only=False)
    payload["__path__"] = CAPTURE
    n_actual = int(payload["num_actual_tokens"])
    device = torch.device("xpu")

    conv_pool, ssm_pool, remap, _ = _build_dense_pool(payload, device)
    sycl_core, _, sycl_conv_post, sycl_ssm_post = _call_sycl(
        payload, conv_pool, ssm_pool, remap, device
    )
    sycl_core = sycl_core[:n_actual].detach().to("cpu").float()
    sycl_conv_post = sycl_conv_post.detach().to("cpu").float().clone()
    sycl_ssm_post = sycl_ssm_post.detach().to("cpu").float().clone()

    # rebuild a fresh pool for the oracle (otherwise it would diff against the
    # SYCL-mutated pool)
    conv_pool2, ssm_pool2, remap2, _ = _build_dense_pool(payload, device)
    oracle_core, oracle_conv_post, oracle_ssm_post, _ = _compute_fla_spec_oracle(
        payload, device
    )
    oracle_core = oracle_core[:n_actual].detach().to("cpu").float()
    oracle_conv_post = oracle_conv_post.detach().to("cpu").float().clone()
    oracle_ssm_post = oracle_ssm_post.detach().to("cpu").float().clone()

    assert sycl_core.shape == oracle_core.shape, (sycl_core.shape, oracle_core.shape)
    print(f"shape (T, H, V) = {tuple(sycl_core.shape)}")

    diff = (sycl_core - oracle_core).abs()
    print(
        f"max abs diff = {diff.max().item():.4e}, "
        f"mean abs diff = {diff.mean().item():.4e}"
    )

    flat_idx = int(diff.argmax().item())
    T, H, V = sycl_core.shape
    t_max, h_max, v_max = (
        flat_idx // (H * V),
        (flat_idx // V) % H,
        flat_idx % V,
    )
    print(
        f"argmax (t, h, v) = ({t_max}, {h_max}, {v_max})  "
        f"sycl={sycl_core[t_max, h_max, v_max].item():.4f}  "
        f"fla={oracle_core[t_max, h_max, v_max].item():.4f}  "
        f"diff={diff[t_max, h_max, v_max].item():.4f}"
    )

    print("\n--- top-20 cells by abs diff ---")
    top_vals, top_idx = diff.flatten().topk(20)
    for v, fi in zip(top_vals.tolist(), top_idx.tolist()):
        t = fi // (H * V)
        h = (fi // V) % H
        vv = fi % V
        print(
            f"  (t={t:2d}, h={h:2d}, v={vv:3d})  "
            f"sycl={sycl_core[t, h, vv].item():+.4f}  "
            f"fla={oracle_core[t, h, vv].item():+.4f}  "
            f"diff={v:+.4f}"
        )

    print("\n--- per-token (t) max diff ---")
    for t in range(T):
        sub = diff[t]
        flat = int(sub.argmax().item())
        h, v = flat // V, flat % V
        print(
            f"  t={t}: max={sub.max().item():.4e} "
            f"@ (h={h}, v={v}), mean={sub.mean().item():.4e}, "
            f"#cells>2e-2: {(sub > 2e-2).sum().item()}"
        )

    print("\n--- per-head (h) max diff (top-10) ---")
    h_max_per = diff.amax(dim=(0, 2))
    top_h = h_max_per.topk(10)
    for v, h in zip(top_h.values.tolist(), top_h.indices.tolist()):
        print(f"  h={h:2d}: max={v:.4e}")

    print("\n--- per-v_dim (v) max diff (top-10) ---")
    v_max_per = diff.amax(dim=(0, 1))
    top_v = v_max_per.topk(10)
    for val, vv in zip(top_v.values.tolist(), top_v.indices.tolist()):
        print(f"  v={vv:3d}: max={val:.4e}")

    # Per-token state diff: each of the 4 spec tokens writes its post-state
    # to spec_state_indices[batch, t]. Diff per-slot.
    spec_idx = payload["spec_state_indices_tensor"]  # raw indices, before remap
    from tests.kernels.xpu.test_spec_gdn_replay import _remap_index_tensor
    spec_idx_dense = _remap_index_tensor(spec_idx, remap).to(torch.long)

    # CONV state diff
    print("\n--- per-spec-token conv_state max abs diff ---")
    print(f"  shape sycl_conv_post = {tuple(sycl_conv_post.shape)}")
    n_seq_c = spec_idx_dense.shape[0]
    K_c = spec_idx_dense.shape[1]
    for batch in range(n_seq_c):
        for t in range(K_c):
            slot = int(spec_idx_dense[batch, t].item())
            if slot <= 0 or slot >= sycl_conv_post.shape[0]:
                continue
            sycl_slot = sycl_conv_post[slot]
            oracle_slot = oracle_conv_post[slot]
            d = (sycl_slot - oracle_slot).abs()
            print(
                f"  batch={batch} t={t} slot={slot}: "
                f"max={d.max().item():.4e}, mean={d.mean().item():.4e}, "
                f"#cells>2e-2: {(d > 2e-2).sum().item()}"
            )
    print("\n--- per-spec-token ssm_state max abs diff ---")
    print(f"  shape sycl_ssm_post = {tuple(sycl_ssm_post.shape)}")
    print(f"  shape oracle_ssm_post = {tuple(oracle_ssm_post.shape)}")
    print(f"  spec_idx_dense = {spec_idx_dense.tolist()}")
    n_seq = spec_idx_dense.shape[0]
    K = spec_idx_dense.shape[1]
    for batch in range(n_seq):
        for t in range(K):
            slot = int(spec_idx_dense[batch, t].item())
            if slot <= 0:
                continue
            sycl_slot = sycl_ssm_post[slot]
            oracle_slot = oracle_ssm_post[slot]
            d = (sycl_slot - oracle_slot).abs()
            print(
                f"  batch={batch} t={t} slot={slot}: "
                f"max={d.max().item():.4e}, mean={d.mean().item():.4e}, "
                f"#cells>2e-2: {(d > 2e-2).sum().item()}"
            )


if __name__ == "__main__":
    main()

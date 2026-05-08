"""Tick-17 zero-state probe.

Hypothesis split (tick 13 Plan A): if SSM-state at the load slot is
zero, the kernel's iter-0 compute reduces to
    state_post_decay = 0
    kv_mem           = 0
    delta            = v_t * beta
    state_post       = (v_t * beta).unsqueeze(-1) * k_norm.unsqueeze(1)
    core_attn_out    = (state_post * q_norm).sum(k)

That eliminates the load and decay paths. Whatever divergence remains
between SYCL and the inline FLA oracle is downstream (rank-1 update,
reduce_over_group, writeback). If divergence vanishes, the bug is in
the load or decay.

Layer-0 K=4 fully-accepted capture (worst layer per tick 12), v=79
hot lane (universal across heads per tick 15).

Run:
  nix develop ~/Projects/vllm-xpu-kernels -c \\
    vllm-run /opt/venv/bin/python /workspace/vllm/.spec-gdn-zerostate-probe.py
"""

from __future__ import annotations

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

CAPTURE = (
    "/tmp/spec_gdn_captures/"
    "tuple_language_model_model_layers_0_linear_attn_000004_spec_K4_min4_max4.pt"
)


def diff_summary(label, sycl, fla, n_actual):
    sycl_c = sycl[:n_actual].detach().to("cpu").float()
    fla_c = fla[:n_actual].detach().to("cpu").float()
    delta = (sycl_c - fla_c).abs()
    over = (delta > 2e-2).sum().item()
    total = delta.numel()
    print(
        f"[{label}] max={delta.max().item():.4e} "
        f"mean={delta.mean().item():.4e} "
        f"fail={over}/{total} ({100*over/total:.3f}%)"
    )
    return delta


def by_vdim(delta, label):
    by_v = delta.amax(dim=(0, 1))
    fail_v = (delta > 2e-2).sum(dim=(0, 1))
    top = torch.topk(by_v, 8)
    print(f"[{label}] top-8 v_dim by max: ", end="")
    print(", ".join(f"v={int(i)}({v:.3f},f={int(fail_v[i])})"
                    for v, i in zip(top.values.tolist(), top.indices.tolist())))


def main():
    payload = torch.load(CAPTURE, map_location="cpu", weights_only=False)
    payload["__path__"] = CAPTURE
    n_actual = int(payload["num_actual_tokens"])
    device = torch.device("xpu")

    print("=" * 72)
    print("Run A: BASELINE (real captured initial state) — for reference")
    print("=" * 72)
    conv_a, ssm_a, remap_a, _ = _build_dense_pool(payload, device)
    sycl_core_a, _, _, _ = _call_sycl(payload, conv_a, ssm_a, remap_a, device)
    fla_core_a, _, _, _ = _compute_fla_spec_oracle(payload, device)
    delta_a = diff_summary("baseline core", sycl_core_a, fla_core_a, n_actual)
    by_vdim(delta_a, "baseline core")
    print(
        f"[baseline core] v=79 max: "
        f"{delta_a[..., 79].max().item():.4e}, "
        f"fail: {int((delta_a[..., 79] > 2e-2).sum().item())}"
    )

    print()
    print("=" * 72)
    print("Run B: ZEROED ssm_state_pre across ALL pool slots")
    print("=" * 72)
    payload_zero = dict(payload)
    payload_zero["ssm_state_pre"] = torch.zeros_like(payload["ssm_state_pre"])

    conv_b, ssm_b, remap_b, _ = _build_dense_pool(payload_zero, device)
    sycl_core_b, _, _, _ = _call_sycl(payload_zero, conv_b, ssm_b, remap_b, device)
    fla_core_b, _, _, _ = _compute_fla_spec_oracle(payload_zero, device)
    delta_b = diff_summary("zerostate core", sycl_core_b, fla_core_b, n_actual)
    by_vdim(delta_b, "zerostate core")
    print(
        f"[zerostate core] v=79 max: "
        f"{delta_b[..., 79].max().item():.4e}, "
        f"fail: {int((delta_b[..., 79] > 2e-2).sum().item())}"
    )

    print()
    print("=" * 72)
    print("Verdict")
    print("=" * 72)
    a_max_v79 = delta_a[..., 79].max().item()
    b_max_v79 = delta_b[..., 79].max().item()
    if b_max_v79 < 0.05 and a_max_v79 > 0.5:
        print(
            f"v=79 fault disappears with zero init "
            f"(baseline max {a_max_v79:.3f} -> zerostate {b_max_v79:.3f}). "
            "Bug is in LOAD or DECAY (lines 144-149 / 167-168)."
        )
    elif b_max_v79 > 0.5:
        print(
            f"v=79 fault persists with zero init "
            f"(baseline {a_max_v79:.3f} -> zerostate {b_max_v79:.3f}). "
            "Bug is downstream of decay: rank-1 update / reduce_over_group / "
            "writeback (lines 207-285)."
        )
    else:
        print(
            f"Inconclusive: baseline v=79 max {a_max_v79:.3f}, "
            f"zerostate v=79 max {b_max_v79:.3f}. Read full output."
        )


if __name__ == "__main__":
    sys.exit(main() or 0)

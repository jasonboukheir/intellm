"""Tick-18 no-op-decay probe.

Tick 17 split the v=79 fault into ~94% load/decay + ~6% downstream. This
script narrows load vs decay by setting `A_log = -100` (so `exp(A_log) ≈ 0`,
hence `g = exp(-exp(A_log) * softplus(a+dt_bias)) ≡ 1`).

Logic:
- If forcing g≡1 with the real captured init state collapses the v=79
  fault → decay (`state_local *= g`, lines 167-168) is the culprit.
- If v=79 still fails ~baseline (~1.0+) → load (lines 144-149) is the
  culprit. The wrong state is loaded; not multiplying it by g doesn't
  fix the wrongness.

Both pyref and SYCL/FLA receive the SAME modified A_log, so any
divergence is attributable to the kernel's load path under same-g.
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
    print("Run A: BASELINE (real A_log, real init state) — for reference")
    print("=" * 72)
    conv_a, ssm_a, remap_a, _ = _build_dense_pool(payload, device)
    sycl_core_a, _, _, _ = _call_sycl(payload, conv_a, ssm_a, remap_a, device)
    fla_core_a, _, _, _ = _compute_fla_spec_oracle(payload, device)
    delta_a = diff_summary("baseline core", sycl_core_a, fla_core_a, n_actual)
    by_vdim(delta_a, "baseline core")
    print(
        f"[baseline core] v=79 max: {delta_a[..., 79].max().item():.4e}, "
        f"fail: {int((delta_a[..., 79] > 2e-2).sum().item())}"
    )

    print()
    print("=" * 72)
    print("Run C: g≡1 (A_log = -100), real init state")
    print("=" * 72)
    payload_g1 = dict(payload)
    A_log_orig = payload["A_log"]
    A_log_g1 = torch.full_like(A_log_orig, -100.0)
    payload_g1["A_log"] = A_log_g1

    conv_c, ssm_c, remap_c, _ = _build_dense_pool(payload_g1, device)
    sycl_core_c, _, _, _ = _call_sycl(payload_g1, conv_c, ssm_c, remap_c, device)
    fla_core_c, _, _, _ = _compute_fla_spec_oracle(payload_g1, device)
    delta_c = diff_summary("g≡1 core", sycl_core_c, fla_core_c, n_actual)
    by_vdim(delta_c, "g≡1 core")
    print(
        f"[g≡1 core] v=79 max: {delta_c[..., 79].max().item():.4e}, "
        f"fail: {int((delta_c[..., 79] > 2e-2).sum().item())}"
    )

    # Sanity check: with A_log=-100, both SYCL and pyref's effective
    # g should be 1.0 ± denormal noise. Confirm by re-running pyref with
    # the modified A_log and inspecting g for head 14.
    A = A_log_g1[14].float()
    g_check = torch.exp(-torch.exp(A) * torch.log1p(torch.exp(torch.tensor(0.0))))
    print(f"[sanity] g[head=14] with A_log=-100: {g_check.item():.10f}")

    print()
    print("=" * 72)
    print("Verdict")
    print("=" * 72)
    a_max_v79 = delta_a[..., 79].max().item()
    c_max_v79 = delta_c[..., 79].max().item()
    if c_max_v79 < 0.1 and a_max_v79 > 0.5:
        print(
            f"v=79 fault disappears with g≡1 "
            f"(baseline {a_max_v79:.3f} -> g≡1 {c_max_v79:.3f}). "
            "Bug is in DECAY (lines 167-168 / state_local *= g). The state "
            "load itself is fine; the multiply by g at j=3 codegens wrong."
        )
    elif c_max_v79 > 0.5:
        print(
            f"v=79 fault persists with g≡1 "
            f"(baseline {a_max_v79:.3f} -> g≡1 {c_max_v79:.3f}). "
            "Bug is in LOAD (lines 144-149). state_local at j=3 of v=79 "
            "loads the wrong value from ssm_state."
        )
    else:
        print(
            f"Partial drop: baseline v=79 max {a_max_v79:.3f}, "
            f"g≡1 v=79 max {c_max_v79:.3f}. Possibly mixed (some load, "
            "some decay) — read full output."
        )


if __name__ == "__main__":
    sys.exit(main() or 0)

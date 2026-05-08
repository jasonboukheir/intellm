"""For the layer-0 rung-4 capture, find which (if any) v_neighbor
makes pyref's iter-0 state_post at (head=14, v_neighbor, k=*) match
SYCL's iter-0 state_post at (head=14, v=79, k=*).

If any v matches within bf16 noise → SYCL is mis-mixing v_neighbor's
lane into v=79's output. The matching v names the off-by-one /
coordinate-mix bug location.

Run inside vllm-dev:
  vllm-run /opt/venv/bin/python /workspace/vllm/.spec-gdn-iter0-vmatch.py
"""
from __future__ import annotations

import math
import sys

import torch

import vllm._xpu_ops  # noqa: F401
import vllm_xpu_kernels._xpu_C  # noqa: F401

sys.path.insert(0, "/workspace/vllm")
from tests.kernels.xpu.test_spec_gdn_replay import (
    _build_dense_pool,
    _call_sycl,
    _compute_fla_spec_oracle,
)

CAPTURE = (
    "/tmp/spec_gdn_captures/"
    "tuple_language_model_model_layers_0_linear_attn_000004_spec_K4_min4_max4.pt"
)


def softplus_t(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x < 20.0, torch.log1p(torch.exp(x)), x)


def pyref_iter0(q, k, v, b, a, A_log, dt_bias, state_init, t=0):
    """Run a SINGLE chunk iteration starting from state_init (fp32, shape
    (H, V, K)). Returns the post-update state (H, V, K) and the per-cell
    core_attn_out (H, V).
    """
    num_v_heads, head_v_dim, head_k_dim = state_init.shape
    num_k_heads = k.shape[1]
    kv_ratio = num_v_heads // num_k_heads
    eps = 1e-6
    scale = 1.0 / math.sqrt(head_k_dim)

    A_log_neg_exp = -torch.exp(A_log)
    dt_bias_f = dt_bias.float()
    kh_per_vh = torch.arange(num_v_heads) // kv_ratio

    b_t = b[t].float()
    beta = torch.sigmoid(b_t)
    a_t = a[t].float() + dt_bias_f
    g = torch.exp(A_log_neg_exp * softplus_t(a_t))

    q_t = q[t].float()
    k_t = k[t].float()
    qs = (q_t * q_t).sum(-1) + eps
    ks = (k_t * k_t).sum(-1) + eps
    q_norm = (q_t / qs.sqrt().unsqueeze(-1)) * scale
    k_norm = k_t / ks.sqrt().unsqueeze(-1)
    q_norm = q_norm[kh_per_vh]
    k_norm = k_norm[kh_per_vh]
    v_t = v[t].float()

    state = state_init.clone() * g.view(num_v_heads, 1, 1)
    kv_mem = torch.einsum("hvk,hk->hv", state, k_norm)
    delta = (v_t - kv_mem) * beta.unsqueeze(-1)
    state = state + delta.unsqueeze(-1) * k_norm.unsqueeze(1)
    res = torch.einsum("hvk,hk->hv", state, q_norm)
    return state, res


def main():
    p = torch.load(CAPTURE, map_location="cpu", weights_only=False)
    cfg = p["layer_config"]
    n_actual = int(p["num_actual_tokens"])
    num_v_heads = cfg["num_v_heads"]
    num_k_heads = cfg["num_k_heads"]
    head_k_dim = cfg["head_k_dim"]
    head_v_dim = cfg["head_v_dim"]
    key_dim = cfg["key_dim"]
    activation = cfg["activation"]

    device = torch.device("xpu")
    p2 = dict(p)
    p2["__path__"] = "vmatch"

    # Build dense pools, run SYCL once.
    conv_pool, ssm_pool, remap, _ = _build_dense_pool(p2, device)
    _, _, _, sycl_ssm_post = _call_sycl(p2, conv_pool, ssm_pool, remap, device)

    # Get post-conv q/k/v from FLA's conv1d (same recipe as the oracle).
    spec_idx_tensor = p["spec_state_indices_tensor"].to(device).to(torch.int32)
    num_acc = p["num_accepted_tokens"].to(device).to(torch.int32)
    spec_qsl = p["spec_query_start_loc"].to(device).to(torch.int32)
    spec_idx_dense = remap.to(device)[spec_idx_tensor.to(torch.long)].to(torch.int32)

    conv_w = p["conv_weight"]
    if conv_w.dim() == 3:
        conv_w = conv_w.view(conv_w.size(0), conv_w.size(2))
    conv_w = conv_w.to(device)

    conv_pool_fresh, ssm_pool_fresh_dev, _, _ = _build_dense_pool(p2, device)
    conv_pool_for_fla = conv_pool_fresh.transpose(-1, -2).contiguous()
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_update,
    )
    mqp = p["mixed_qkv"][:n_actual].to(device).clone()
    mqp = causal_conv1d_update(
        mqp, conv_pool_for_fla, conv_w, p["conv_bias"], activation,
        conv_state_indices=spec_idx_dense[:, 0][:int(p["num_spec_decodes"])].contiguous(),
        num_accepted_tokens=num_acc,
        query_start_loc=spec_qsl,
        max_query_len=spec_idx_dense.size(-1),
        validate_data=False,
    )
    q_post = mqp[:, :key_dim].view(n_actual, num_k_heads, head_k_dim).cpu()
    k_post = mqp[:, key_dim:2 * key_dim].view(n_actual, num_k_heads, head_k_dim).cpu()
    v_post = mqp[:, 2 * key_dim:].view(n_actual, num_v_heads, head_v_dim).cpu()

    # Initial state: load_slot is spec_idx_dense[0, num_acc[0]-1] in the dense pool.
    n0 = int(num_acc[0].item())
    load_slot = int(spec_idx_dense[0, n0 - 1].item())
    state_init = ssm_pool_fresh_dev[load_slot].cpu().float().clone()
    print(f"[vmatch] load_slot dense={load_slot}, "
          f"|state_init|={state_init.abs().max().item():.4e}, "
          f"shape={tuple(state_init.shape)}")

    # SYCL iter-0 state_post is at slot spec_idx_dense[0, 0].
    iter0_slot = int(spec_idx_dense[0, 0].item())
    sycl_iter0 = sycl_ssm_post[iter0_slot].cpu().float()  # (H, V, K)
    print(f"[vmatch] iter0_slot dense={iter0_slot}, "
          f"|sycl iter0|={sycl_iter0.abs().max().item():.4e}")

    # Run pyref's iter-0 starting from the FULL captured state_init.
    # (No per-v swap — this gives us the correct pyref iter-0 state.)
    pyref_iter0_state, pyref_iter0_core = pyref_iter0(
        q_post, k_post, v_post,
        p["b"][:n_actual].cpu(), p["a"][:n_actual].cpu(),
        p["A_log"].cpu(), p["dt_bias"].cpu(),
        state_init, t=0,
    )

    target_h = 14
    target_v = 79
    sycl_target = sycl_iter0[target_h, target_v]  # (K,)

    print(f"\n[vmatch] target: SYCL iter-0 state at "
          f"(head={target_h}, v={target_v}), |x|={sycl_target.abs().max().item():.3e}")
    print(f"[vmatch] pyref same coord: |x|="
          f"{pyref_iter0_state[target_h, target_v].abs().max().item():.3e}, "
          f"diff={(pyref_iter0_state[target_h, target_v] - sycl_target).abs().max().item():.3e}")

    # For each v_candidate ∈ 0..127, check distance of pyref(head=14, v_candidate)
    # to SYCL(head=14, v=79). Smallest distance names the coordinate mix.
    print("\n=== pyref(h=14, v=v_cand) vs sycl(h=14, v=79) — top-10 closest v_cand ===")
    distances = []
    for v_cand in range(head_v_dim):
        d = (pyref_iter0_state[target_h, v_cand] - sycl_target).abs().max().item()
        distances.append((d, v_cand))
    distances.sort()
    for d, v_cand in distances[:10]:
        marker = "  <-- target" if v_cand == target_v else ""
        print(f"  v_cand={v_cand:>3} max_diff={d:>10.4e}{marker}")

    # Also check across ALL heads — maybe SYCL is reading from a different
    # head's same-v lane.
    print("\n=== pyref(h=h_cand, v=79) vs sycl(h=14, v=79) — top-10 closest h_cand ===")
    h_distances = []
    for h_cand in range(num_v_heads):
        d = (pyref_iter0_state[h_cand, target_v] - sycl_target).abs().max().item()
        h_distances.append((d, h_cand))
    h_distances.sort()
    for d, h_cand in h_distances[:10]:
        marker = "  <-- target" if h_cand == target_h else ""
        print(f"  h_cand={h_cand:>3} max_diff={d:>10.4e}{marker}")

    # And across (h, v) joint space — top-10 closest matches anywhere.
    print("\n=== pyref(h_cand, v_cand) vs sycl(h=14, v=79) — top-10 anywhere ===")
    flat = []
    for h_cand in range(num_v_heads):
        for v_cand in range(head_v_dim):
            d = (pyref_iter0_state[h_cand, v_cand] - sycl_target).abs().max().item()
            flat.append((d, h_cand, v_cand))
    flat.sort()
    for d, h_cand, v_cand in flat[:10]:
        marker = "  <-- target" if (h_cand, v_cand) == (target_h, target_v) else ""
        print(f"  h={h_cand:>3} v={v_cand:>3} max_diff={d:>10.4e}{marker}")

    # Same probe on the INITIAL state — does sycl's iter-0 v=79 output look
    # like a *neighbor* of state_init's v=79 (i.e., maybe the load itself
    # is correct but the kernel skips the iter-0 update for some lanes)?
    print("\n=== state_init(h=14, v=v_cand) vs sycl iter-0(h=14, v=79) — top-10 ===")
    sd = []
    for v_cand in range(head_v_dim):
        d = (state_init[target_h, v_cand] - sycl_target).abs().max().item()
        sd.append((d, v_cand))
    sd.sort()
    for d, v_cand in sd[:10]:
        marker = "  <-- target" if v_cand == target_v else ""
        print(f"  init v_cand={v_cand:>3} max_diff={d:>10.4e}{marker}")


if __name__ == "__main__":
    sys.exit(main() or 0)

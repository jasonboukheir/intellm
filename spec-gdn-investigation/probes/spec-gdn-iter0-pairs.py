"""Enumerate all iter-0 (h, v) cells where SYCL diverges from pyref for
the layer-0 rung-4 capture, and look for structural pairings:
  - even-odd v_head pairs (h, h^1) co-failing at the same v?
  - v-pairs (v, v^1) co-failing at the same head?
  - cluster shape across (h, v) — sub-group / lane / bucket?

Run inside vllm-dev:
  vllm-run /opt/venv/bin/python /workspace/vllm/.spec-gdn-iter0-pairs.py
"""
from __future__ import annotations

import math
import sys

import torch

import vllm._xpu_ops  # noqa: F401
import vllm_xpu_kernels._xpu_C  # noqa: F401

sys.path.insert(0, "/workspace/vllm")
from tests.kernels.xpu.test_spec_gdn_replay import _build_dense_pool, _call_sycl

CAPTURE = (
    "/tmp/spec_gdn_captures/"
    "tuple_language_model_model_layers_0_linear_attn_000004_spec_K4_min4_max4.pt"
)


def softplus_t(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x < 20.0, torch.log1p(torch.exp(x)), x)


def pyref_iter0_full(q, k, v, b, a, A_log, dt_bias, state_init, t=0):
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
    return state  # (H, V, K) fp32


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
    p2["__path__"] = "pairs"

    conv_pool, ssm_pool, remap, _ = _build_dense_pool(p2, device)
    _, _, _, sycl_ssm_post = _call_sycl(p2, conv_pool, ssm_pool, remap, device)

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

    n0 = int(num_acc[0].item())
    load_slot = int(spec_idx_dense[0, n0 - 1].item())
    iter0_slot = int(spec_idx_dense[0, 0].item())
    state_init = ssm_pool_fresh_dev[load_slot].cpu().float()
    sycl_iter0 = sycl_ssm_post[iter0_slot].cpu().float()  # (H, V, K)

    pyref_iter0 = pyref_iter0_full(
        q_post, k_post, v_post,
        p["b"][:n_actual].cpu(), p["a"][:n_actual].cpu(),
        p["A_log"].cpu(), p["dt_bias"].cpu(),
        state_init, t=0,
    )
    delta = (sycl_iter0 - pyref_iter0).abs()  # (H, V, K)
    tol = 2e-2 + 2e-2 * pyref_iter0.abs()
    fail = delta > tol
    fail_per_hv = fail.sum(dim=2)  # (H, V) — number of failing k per (h, v)
    max_per_hv = delta.max(dim=2).values  # (H, V)

    print(f"=== iter-0 failure mask, layer-0 capture ===")
    print(f"total fails: {int(fail.sum())} / {fail.numel()} cells "
          f"({100*fail.float().mean():.4f}%)")
    print(f"|sycl-pyref| max abs: {delta.max().item():.4e}, "
          f"mean: {delta.mean().item():.4e}")

    # Top 30 (h, v) by max_per_hv
    flat = []
    for h in range(num_v_heads):
        for vv in range(head_v_dim):
            if max_per_hv[h, vv].item() > 2e-2:
                flat.append((max_per_hv[h, vv].item(),
                             int(fail_per_hv[h, vv].item()), h, vv))
    flat.sort(reverse=True)

    print(f"\n=== top 30 (h, v) by max-cell error (max>2e-2) — total such cells: {len(flat)} ===")
    for max_v, fc, h, vv in flat[:30]:
        print(f"  h={h:>2} v={vv:>3} max_diff={max_v:>9.3f} "
              f"fail_k={fc:>3}/{head_k_dim}")

    # Even-odd v_head pair check
    print("\n=== even-odd v_head pairing (h, h^1) ===")
    pair_evidence = []
    for ph in range(0, num_v_heads, 2):
        h_lo, h_hi = ph, ph + 1
        for vv in range(head_v_dim):
            if max_per_hv[h_lo, vv] > 2e-2 and max_per_hv[h_hi, vv] > 2e-2:
                pair_evidence.append((
                    max(max_per_hv[h_lo, vv].item(), max_per_hv[h_hi, vv].item()),
                    h_lo, h_hi, vv,
                    max_per_hv[h_lo, vv].item(),
                    max_per_hv[h_hi, vv].item(),
                ))
    pair_evidence.sort(reverse=True)
    print(f"head-pairs (h, h^1) where BOTH halves fail at same v: {len(pair_evidence)}")
    for max_v, h_lo, h_hi, vv, lo, hi in pair_evidence[:15]:
        ratio = lo / hi if hi > 0 else 0
        print(f"  ({h_lo:>2}, {h_hi:>2}) at v={vv:>3}: "
              f"lo={lo:>8.3f} hi={hi:>8.3f} ratio={ratio:.3f}")

    # Compare vs all (h, v) fails: how many fails are at paired heads vs solo?
    paired_count = 0
    solo_count = 0
    for h in range(num_v_heads):
        for vv in range(head_v_dim):
            if max_per_hv[h, vv] > 2e-2:
                pair_h = h ^ 1
                if max_per_hv[pair_h, vv] > 2e-2:
                    paired_count += 1
                else:
                    solo_count += 1
    print(f"\nfails at paired heads (both h and h^1 fail at same v): {paired_count}")
    print(f"fails solo (only one of h, h^1 fails at this v):           {solo_count}")

    # v-pair check at the same head: (v, v^1) co-failing
    print("\n=== v-pair (v, v^1) co-failing at same head ===")
    v_pair_evidence = []
    for h in range(num_v_heads):
        for vv in range(0, head_v_dim, 2):
            v_lo, v_hi = vv, vv + 1
            if max_per_hv[h, v_lo] > 2e-2 and max_per_hv[h, v_hi] > 2e-2:
                v_pair_evidence.append((
                    max(max_per_hv[h, v_lo].item(), max_per_hv[h, v_hi].item()),
                    h, v_lo, v_hi,
                    max_per_hv[h, v_lo].item(),
                    max_per_hv[h, v_hi].item(),
                ))
    v_pair_evidence.sort(reverse=True)
    print(f"v-pairs (v, v^1) where BOTH halves fail at same head: "
          f"{len(v_pair_evidence)}")
    for max_v, h, v_lo, v_hi, lo, hi in v_pair_evidence[:10]:
        print(f"  h={h:>2} (v={v_lo:>3}, v={v_hi:>3}): "
              f"lo={lo:>8.3f} hi={hi:>8.3f}")

    # Per-failing-(h, v) breakdown by v-mod-4 (j-lane)
    print("\n=== per-j (v % 4) breakdown of failing (h, v) cells ===")
    j_count = [0, 0, 0, 0]
    for h in range(num_v_heads):
        for vv in range(head_v_dim):
            if max_per_hv[h, vv] > 2e-2:
                j_count[vv % 4] += 1
    for j in range(4):
        print(f"  j={j} (v%4=={j}): {j_count[j]} failing (h, v) cells")

    # v_bucket (v // 32) breakdown
    print("\n=== per-v_bucket (v // 32) breakdown ===")
    bucket_count = [0, 0, 0, 0]
    for h in range(num_v_heads):
        for vv in range(head_v_dim):
            if max_per_hv[h, vv] > 2e-2:
                bucket_count[vv // 32] += 1
    for b in range(4):
        print(f"  bucket={b} (v={32*b}..{32*b+31}): {bucket_count[b]} cells")

    # Heads that have at least one fail
    heads_with_fails = [h for h in range(num_v_heads)
                        if int((max_per_hv[h] > 2e-2).sum()) > 0]
    print(f"\nheads with any iter-0 fail: {heads_with_fails}")
    print(f"  (kh_per_vh maps to k_head = h // 8, so 0..7→0, 8..15→1, "
          f"16..23→2, 24..31→3)")


if __name__ == "__main__":
    sys.exit(main() or 0)

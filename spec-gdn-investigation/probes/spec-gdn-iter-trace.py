"""Iteration-by-iteration trace of state_local at the layer-0 v=79 locus
to diagnose whether the SYCL codegen bug is a per-iteration error
(diverges at iter 0) or bf16 accumulation drift (diverges at iter >=1).

Approach: rung-4 capture has K=4 spec slots; the kernel's per-token
writeback puts post-iter-t state into spec_idx[batch, t]. So a SINGLE
SYCL call gives us all 4 iteration outputs (slots 0..3 of the spec
window). Compare each iter's slot, head, v=79 against the same iter
of a pyref_hpp that traces state_local in fp32.

Run inside vllm-dev:
  vllm-run /opt/venv/bin/python /workspace/vllm/.spec-gdn-iter-trace.py

No kernel rebuild needed.
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


def pyref_hpp_traced(q, k, v, b, a, A_log, dt_bias, ssm_pool_init, qsl,
                     spec_idx, num_acc, watch_head, watch_v):
    """Pure fp32 transcription of gated_delta_rule.hpp's chunk loop.

    Returns:
      core_attn_out  (T, H, V) fp32
      out_pool       (slot, H, V, K) fp32  (one slot per spec iteration)
      trace          list[dict] one per iter t, with the full state_local
                     slice at watch_head, watch_v across all k.
    """
    T_total, num_v_heads, head_v_dim = v.shape
    num_k_heads = k.shape[1]
    head_k_dim = k.shape[2]
    kv_ratio = num_v_heads // num_k_heads
    eps = 1e-6
    scale = 1.0 / math.sqrt(head_k_dim)

    out_pool = ssm_pool_init.clone()
    core = torch.zeros(
        T_total, num_v_heads, head_v_dim, dtype=v.dtype, device=v.device
    )

    A_log_neg_exp = -torch.exp(A_log)
    dt_bias_f = dt_bias.float()
    kh_per_vh = torch.arange(num_v_heads, device=v.device) // kv_ratio

    trace = []

    for batch in range(qsl.numel() - 1):
        s = int(qsl[batch].item())
        e = int(qsl[batch + 1].item())
        if e <= s:
            continue
        n = int(num_acc[batch].item())
        if n <= 0:
            continue
        load_slot = int(spec_idx[batch, n - 1].item())
        if load_slot <= 0:
            continue
        state = out_pool[load_slot].clone().float()
        # state shape: (H, V, K) — already in (head, v, k) layout per the
        # tick-4 axis correction.

        for t in range(s, e):
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

            state_pre = state.clone()  # before decay
            state = state * g.view(num_v_heads, 1, 1)
            state_post_decay = state.clone()
            kv_mem = torch.einsum("hvk,hk->hv", state, k_norm)
            delta = (v_t - kv_mem) * beta.unsqueeze(-1)
            state = state + delta.unsqueeze(-1) * k_norm.unsqueeze(1)
            state_post_update = state.clone()

            res = torch.einsum("hvk,hk->hv", state, q_norm)
            core[t] = res.to(core.dtype)

            # snapshot watched cell
            trace.append(dict(
                t=t - s,
                head=watch_head,
                v=watch_v,
                state_pre=state_pre[watch_head, watch_v].clone(),
                state_post_decay=state_post_decay[watch_head, watch_v].clone(),
                kv_mem=kv_mem[watch_head, watch_v].item(),
                delta=delta[watch_head, watch_v].item(),
                state_post_update=state_post_update[watch_head, watch_v].clone(),
                core_at_cell=res[watch_head, watch_v].item(),
                g_h=g[watch_head].item(),
                beta_h=beta[watch_head].item(),
                v_t_at_cell=v_t[watch_head, watch_v].item(),
                k_norm=k_norm[watch_head].clone(),
                q_norm=q_norm[watch_head].clone(),
            ))

            spec_slot = int(spec_idx[batch, t - s].item())
            if spec_slot > 0:
                out_pool[spec_slot] = state.to(out_pool.dtype)

    return core, out_pool, trace


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

    # ---- Build dense pools and remap (matches test) ----
    p2 = dict(p)
    p2["__path__"] = "iter-trace"
    conv_pool, ssm_pool, remap, _ = _build_dense_pool(p2, device)

    # Find watch_head from layer-0 rung-4: tick 12 reports v=79 dominates.
    # Pick the worst-failing (token, head, v=79) cell.
    sycl_core, _, _, sycl_ssm_post = _call_sycl(p2, conv_pool, ssm_pool, remap, device)

    # We need the FLA oracle's core to find worst cell.
    from tests.kernels.xpu.test_spec_gdn_replay import _compute_fla_spec_oracle
    fla_core, _, fla_ssm_post, _ = _compute_fla_spec_oracle(p2, device)

    sycl_c = sycl_core[:n_actual].detach().to("cpu").float()
    fla_c = fla_core[:n_actual].detach().to("cpu").float()
    delta_c = (sycl_c - fla_c).abs()

    # restrict to v=79
    delta_v79 = delta_c[..., 79]   # shape (T, H)
    flat = delta_v79.flatten()
    top_idx = int(flat.argmax().item())
    t_idx = top_idx // num_v_heads
    h_idx = top_idx % num_v_heads
    print(
        f"[trace] worst v=79 cell: token={t_idx} head={h_idx} "
        f"sycl={sycl_c[t_idx, h_idx, 79]:.6f} "
        f"fla={fla_c[t_idx, h_idx, 79]:.6f} "
        f"delta={delta_v79[t_idx, h_idx]:.6f}"
    )
    watch_head, watch_v = h_idx, 79

    # ---- Build pyref inputs from captured mixed_qkv via FLA's conv1d ----
    spec_idx_tensor = p["spec_state_indices_tensor"].to(device).to(torch.int32)
    num_acc = p["num_accepted_tokens"].to(device).to(torch.int32)
    spec_qsl = p["spec_query_start_loc"].to(device).to(torch.int32)
    spec_idx_dense = remap.to(device)[spec_idx_tensor.to(torch.long)].to(torch.int32)

    conv_w = p["conv_weight"]
    if conv_w.dim() == 3:
        conv_w = conv_w.view(conv_w.size(0), conv_w.size(2))
    conv_w = conv_w.to(device)

    # Re-build a FRESH conv pool — _call_sycl mutated the previous one in place.
    conv_pool_fresh, _, _, _ = _build_dense_pool(p2, device)
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
    q_post = mqp[:, :key_dim].view(n_actual, num_k_heads, head_k_dim)
    k_post = mqp[:, key_dim:2*key_dim].view(n_actual, num_k_heads, head_k_dim)
    v_post = mqp[:, 2*key_dim:].view(n_actual, num_v_heads, head_v_dim)

    # ---- pyref on CPU fp32 with per-iter trace ----
    _, ssm_pool_pyref_cpu, _, _ = _build_dense_pool(p2, torch.device("cpu"))
    qsl_b = torch.tensor([0, n_actual], dtype=torch.long)
    pyref_core, pyref_pool, trace = pyref_hpp_traced(
        q_post.cpu(), k_post.cpu(), v_post.cpu(),
        p["b"][:n_actual].cpu(), p["a"][:n_actual].cpu(),
        p["A_log"].cpu(), p["dt_bias"].cpu(),
        ssm_pool_pyref_cpu,
        qsl_b,
        spec_idx_dense.to(torch.long).cpu(),
        num_acc.to(torch.long).cpu(),
        watch_head, watch_v,
    )

    # ---- SYCL per-iter state extracted from sycl_ssm_post slots ----
    # Layout: (slot, H, V, K). spec window slot t = spec_idx_dense[0, t].
    print("\n=== Per-iteration trace at layer-0, head=%d, v=%d, all k ===" %
          (watch_head, watch_v))
    print(f"{'iter':>4} | {'g':>10} {'beta':>10} {'v_t':>10} | "
          f"{'pyref|state_post|':>18} {'sycl|state_post|':>18} | "
          f"{'max abs diff':>12} | {'pyref core':>11} {'sycl core':>11}")
    for ev in trace:
        t = ev["t"]
        slot_dense = int(spec_idx_dense[0, t].item())
        sycl_state_t = sycl_ssm_post[slot_dense, watch_head, watch_v].cpu().float()
        pyref_state_t = ev["state_post_update"]
        diff = (sycl_state_t - pyref_state_t).abs()
        sycl_core_at = sycl_c[t, watch_head, watch_v].item()
        print(
            f"{t:>4} | {ev['g_h']:>10.4f} {ev['beta_h']:>10.4f} "
            f"{ev['v_t_at_cell']:>10.4f} | "
            f"{pyref_state_t.abs().max().item():>18.4e} "
            f"{sycl_state_t.abs().max().item():>18.4e} | "
            f"{diff.max().item():>12.4e} | "
            f"{ev['core_at_cell']:>11.4f} {sycl_core_at:>11.4f}"
        )

    # Per-iter k-coordinate breakdown for the worst diff iter
    worst_iter = max(
        range(len(trace)),
        key=lambda i: (
            sycl_ssm_post[
                int(spec_idx_dense[0, trace[i]["t"]].item()), watch_head, watch_v
            ].cpu().float()
            - trace[i]["state_post_update"]
        ).abs().max().item()
    )
    t = trace[worst_iter]["t"]
    slot_dense = int(spec_idx_dense[0, t].item())
    print(f"\n=== K-coord breakdown at worst iter t={t}, head={watch_head}, "
          f"v={watch_v}, slot_dense={slot_dense} ===")
    sycl_k = sycl_ssm_post[slot_dense, watch_head, watch_v].cpu().float()
    pyref_k = trace[worst_iter]["state_post_update"]
    for ki in range(head_k_dim):
        d = abs(sycl_k[ki].item() - pyref_k[ki].item())
        if d > 1e-3:
            print(
                f"  k={ki:>3}: sycl={sycl_k[ki].item():>10.4f} "
                f"pyref={pyref_k[ki].item():>10.4f} d={d:>10.4f}"
            )


if __name__ == "__main__":
    sys.exit(main() or 0)

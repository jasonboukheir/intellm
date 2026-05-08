"""Cross-check three implementations of the same gated_delta_rule algorithm
on identical inputs:

  pyref_hpp        = pure-Python transcription of gated_delta_rule.hpp
  FLA-recurrence   = vllm's actual fused_sigmoid_gating_delta_rule_update
                     Triton kernel
  SYCL native      = the kernel we suspected of having a bug

Inputs: FLA's own causal_conv1d_update output (no torch-conv1d intermediate).

If pyref_hpp ≈ FLA-recurrence ≈ SYCL → the algorithm is correctly implemented
in all three; any divergence vs FLA-captured is explained by something other
than the gated_delta_rule math (e.g. capture-time issue, hidden config).
"""
from __future__ import annotations
import math, sys
import torch

CAPTURE = ("/tmp/spec_gdn_captures/"
           "tuple_language_model_model_layers_0_linear_attn_000004_spec_K4_min4_max4.pt")


def softplus_t(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x < 20.0, torch.log1p(torch.exp(x)), x)


def pyref_hpp(q, k, v, b, a, A_log, dt_bias, ssm_pool, qsl,
              spec_idx, num_acc):
    T_total, num_v_heads, head_v_dim = v.shape
    num_k_heads = k.shape[1]
    head_k_dim = k.shape[2]
    kv_ratio = num_v_heads // num_k_heads
    eps = 1e-6
    scale = 1.0 / math.sqrt(head_k_dim)

    out_pool = ssm_pool.clone()
    core = torch.zeros(T_total, num_v_heads, head_v_dim,
                       dtype=v.dtype, device=v.device)

    A_log_neg_exp = -torch.exp(A_log)
    dt_bias_f = dt_bias.float()
    kh_per_vh = (torch.arange(num_v_heads, device=v.device) // kv_ratio)

    for batch in range(qsl.numel() - 1):
        s = int(qsl[batch].item()); e = int(qsl[batch + 1].item())
        if e <= s:
            continue
        n = int(num_acc[batch].item())
        if n <= 0:
            continue
        load_slot = int(spec_idx[batch, n - 1].item())
        if load_slot <= 0:
            continue
        state = out_pool[load_slot].clone().float()

        for t in range(s, e):
            b_t = b[t].float(); beta = torch.sigmoid(b_t)
            a_t = a[t].float() + dt_bias_f
            g = torch.exp(A_log_neg_exp * softplus_t(a_t))

            q_t = q[t].float(); k_t = k[t].float()
            qs = (q_t * q_t).sum(-1) + eps
            ks = (k_t * k_t).sum(-1) + eps
            q_norm = (q_t / qs.sqrt().unsqueeze(-1)) * scale
            k_norm = k_t / ks.sqrt().unsqueeze(-1)
            q_norm = q_norm[kh_per_vh]
            k_norm = k_norm[kh_per_vh]

            v_t = v[t].float()
            state = state * g.view(num_v_heads, 1, 1)
            kv_mem = torch.einsum("hvk,hk->hv", state, k_norm)
            delta = (v_t - kv_mem) * beta.unsqueeze(-1)
            state = state + delta.unsqueeze(-1) * k_norm.unsqueeze(1)
            res = torch.einsum("hvk,hk->hv", state, q_norm)
            core[t] = res.to(core.dtype)

            spec_slot = int(spec_idx[batch, t - s].item())
            if spec_slot > 0:
                out_pool[spec_slot] = state.to(out_pool.dtype)

    return core, out_pool


def main():
    p = torch.load(CAPTURE, map_location="cpu", weights_only=False)
    cfg = p["layer_config"]
    n_actual = int(p["num_actual_tokens"])
    num_v_heads = cfg["num_v_heads"]; num_k_heads = cfg["num_k_heads"]
    head_k_dim = cfg["head_k_dim"]; head_v_dim = cfg["head_v_dim"]
    key_dim = cfg["key_dim"]; value_dim = cfg["value_dim"]
    activation = cfg["activation"]

    device = torch.device("xpu")
    mixed_qkv = p["mixed_qkv"][:n_actual].to(device)
    conv_w = p["conv_weight"]
    if conv_w.dim() == 3: conv_w = conv_w.view(conv_w.size(0), conv_w.size(2))
    conv_w = conv_w.to(device)

    slots_captured = p["slot_indices"].to(torch.long)
    real_slots = sorted({int(s) for s in slots_captured.tolist()} - {0})
    pool_size = len(real_slots) + 1
    max_slot = int(slots_captured.max().item()) + 1
    remap = torch.zeros(max_slot, dtype=torch.long)
    for i, s in enumerate(real_slots, start=1):
        remap[s] = i

    conv_state_pre = p["conv_state_pre"]
    state_len = conv_state_pre.shape[1]
    dim = conv_state_pre.shape[-1]
    conv_pool = torch.zeros((pool_size, state_len, dim),
                            dtype=conv_state_pre.dtype, device=device)
    slots_dense = remap[slots_captured]
    keep = slots_dense > 0
    conv_pool[slots_dense[keep]] = conv_state_pre[keep].to(device)
    conv_pool_for_fla = conv_pool.transpose(-1, -2).contiguous()

    spec_idx_tensor = p["spec_state_indices_tensor"].to(device).to(torch.int32)
    num_acc = p["num_accepted_tokens"].to(device).to(torch.int32)
    spec_qsl = p["spec_query_start_loc"].to(device).to(torch.int32)
    spec_idx_dense = remap.to(device)[spec_idx_tensor.to(torch.long)].to(torch.int32)

    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_update,
    )
    mqp = mixed_qkv.clone()
    mqp = causal_conv1d_update(
        mqp, conv_pool_for_fla, conv_w, None, activation,
        conv_state_indices=spec_idx_dense[:, 0][:int(p["num_spec_decodes"])],
        num_accepted_tokens=num_acc,
        query_start_loc=spec_qsl,
        max_query_len=spec_idx_tensor.size(-1),
        validate_data=False,
    )

    q = mqp[:, :key_dim].view(n_actual, num_k_heads, head_k_dim)
    k = mqp[:, key_dim:2*key_dim].view(n_actual, num_k_heads, head_k_dim)
    v = mqp[:, 2*key_dim:].view(n_actual, num_v_heads, head_v_dim)

    ssm_pre = p["ssm_state_pre"]
    ssm_pool_base = torch.zeros((pool_size,) + tuple(ssm_pre.shape[1:]),
                                dtype=ssm_pre.dtype, device=device)
    ssm_pool_base[slots_dense[keep]] = ssm_pre[keep].to(device)

    # ---- pyref_hpp on CPU (avoids xpu/cpu mixing) ----
    qsl_b = torch.tensor([0, n_actual], dtype=torch.long)
    pyref_core, pyref_pool = pyref_hpp(
        q.cpu(), k.cpu(), v.cpu(),
        p["b"][:n_actual].cpu(), p["a"][:n_actual].cpu(),
        p["A_log"].cpu(), p["dt_bias"].cpu(),
        ssm_pool_base.cpu(),
        qsl_b,
        p["spec_state_indices_tensor"].to(torch.long).cpu(),
        p["num_accepted_tokens"].to(torch.long).cpu(),
    )
    # pyref operates on the unmapped slot indices; rebuild dense pool
    # for comparison
    # For the comparison, rebuild a dense pool of pyref_pool at remapped slots.
    pyref_dense_pool = torch.zeros_like(ssm_pool_base.cpu())
    for src in real_slots:
        dst = int(remap[src].item())
        pyref_dense_pool[dst] = pyref_pool[src]
    pyref_dense_pool = pyref_dense_pool.to(device)

    # ---- FLA recurrence ----
    from vllm.model_executor.layers.fla.ops import (
        fused_sigmoid_gating_delta_rule_update,
    )
    ssm_pool_fla = ssm_pool_base.clone()
    fla_core, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=p["A_log"].to(device),
        a=p["a"][:n_actual].to(device).unsqueeze(0).contiguous(),
        b=p["b"][:n_actual].to(device).unsqueeze(0).contiguous(),
        dt_bias=p["dt_bias"].to(device),
        q=q.unsqueeze(0).contiguous(),
        k=k.unsqueeze(0).contiguous(),
        v=v.unsqueeze(0).contiguous(),
        initial_state=ssm_pool_fla,
        inplace_final_state=True,
        cu_seqlens=spec_qsl[:int(p["num_spec_decodes"]) + 1].contiguous(),
        ssm_state_indices=spec_idx_dense,
        num_accepted_tokens=num_acc,
        use_qk_l2norm_in_kernel=True,
    )
    fla_core_cpu = fla_core[0, :n_actual].float().cpu()
    fla_pool_cpu = ssm_pool_fla.cpu()

    # ---- SYCL ----
    sys.path.insert(0, "/workspace/vllm")
    from tests.kernels.xpu.test_spec_gdn_replay import _build_dense_pool, _call_sycl
    p2 = dict(p); p2["__path__"] = "x"
    cpool, spool, rmap, _ = _build_dense_pool(p2, device)
    sycl_core, _, _, sycl_pool = _call_sycl(p2, cpool, spool, rmap, device)
    sycl_core_cpu = sycl_core[:n_actual].float().cpu()
    sycl_pool_cpu = sycl_pool.cpu()

    # ---- Cross-diff ----
    pcpu = pyref_core.float().cpu()
    print("=== core_attn_out cross-diff (atol=2e-2 + 2e-2*|ref|) ===")
    def diff(a, b, label):
        d = (a - b).abs()
        tol = 2e-2 + 2e-2 * b.abs()
        f = d > tol
        print(f"{label}: max={d.max().item():.4e} mean={d.mean().item():.4e} "
              f"fail={int(f.sum())}/{f.numel()}")

    diff(pcpu, fla_core_cpu, "pyref vs FLA-recurrence")
    diff(sycl_core_cpu, fla_core_cpu, "SYCL  vs FLA-recurrence")
    diff(pcpu, sycl_core_cpu, "pyref vs SYCL")
    diff(pcpu, p["core_attn_out"][:n_actual].float(), "pyref vs FLA-CAPTURED")
    diff(fla_core_cpu, p["core_attn_out"][:n_actual].float(), "FLA-recurrence vs FLA-CAPTURED")
    diff(sycl_core_cpu, p["core_attn_out"][:n_actual].float(), "SYCL  vs FLA-CAPTURED")

    print("\n=== ssm_state_post cross-diff at captured slots ===")
    pyref_kept = pyref_dense_pool[slots_dense[keep]].cpu()
    fla_kept = fla_pool_cpu[slots_dense[keep]]
    sycl_kept = sycl_pool_cpu[slots_dense[keep]]
    cap_kept = p["ssm_state_post"][keep.cpu()]
    diff(pyref_kept, fla_kept, "pyref vs FLA-recurrence")
    diff(sycl_kept, fla_kept, "SYCL  vs FLA-recurrence")
    diff(pyref_kept, sycl_kept, "pyref vs SYCL")
    diff(pyref_kept, cap_kept, "pyref vs FLA-CAPTURED")
    diff(fla_kept, cap_kept, "FLA-recurrence vs FLA-CAPTURED")
    diff(sycl_kept, cap_kept, "SYCL  vs FLA-CAPTURED")


if __name__ == "__main__":
    sys.exit(main())

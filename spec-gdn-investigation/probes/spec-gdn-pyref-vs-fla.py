"""Disambiguate: is the bug in conv1d, or in the gated_delta_rule math?

Run two tests on the rung-4 capture:
  (A) my torch conv1d's output  vs  FLA's `causal_conv1d_update` output  → conv equality?
  (B) FLA's `fused_sigmoid_gating_delta_rule_update` on my torch-conv outputs
       vs FLA-captured ssm_state_post / core_attn_out.

Outcomes:
  A passes, B passes  → my pyref-of-hpp must have a transcription bug; revisit math.
  A passes, B fails   → algorithm divergence remains; my torch conv1d is right but
                        running FLA's kernel on it doesn't reproduce FLA's captured
                        output (impossible if my conv1d truly matches FLA's input).
  A fails, B fails    → my torch conv1d ≠ FLA's `causal_conv1d_update` → conv1d is
                        the divergence point; pyref of hpp produced same wrong answer
                        as SYCL because both use a different conv1d than FLA.
  A fails, B passes   → strange.

Run inside vllm-dev container (needs XPU for FLA Triton call):
  /opt/venv/bin/python /workspace/vllm/.spec-gdn-pyref-vs-fla.py
"""
from __future__ import annotations
import math, sys
import torch

CAPTURE = ("/tmp/spec_gdn_captures/"
           "tuple_language_model_model_layers_0_linear_attn_000004_spec_K4_min4_max4.pt")


def torch_causal_conv1d(
    x: torch.Tensor,           # [T, C]
    weight: torch.Tensor,      # [C, W]
    bias: torch.Tensor | None,
    conv_state: torch.Tensor,  # [W-1, C]
    activation: str = "silu",
) -> torch.Tensor:
    """Faithful sequential causal conv1d. Output dtype matches input."""
    T, C = x.shape
    W = weight.shape[-1]
    full = torch.cat([conv_state.to(x.dtype), x], dim=0)   # [W-1+T, C]
    w = weight.float()
    out = torch.zeros((T, C), dtype=torch.float32, device=x.device)
    for t in range(T):
        win = full[t:t+W].float()                          # [W, C]
        out[t] = (win * w.t()).sum(dim=0)
    if bias is not None:
        out += bias.float().unsqueeze(0)
    if activation in ("silu", "swish"):
        out = out * torch.sigmoid(out)
    return out.to(x.dtype)


def main():
    p = torch.load(CAPTURE, map_location="cpu", weights_only=False)
    cfg = p["layer_config"]
    n_actual = int(p["num_actual_tokens"])
    num_v_heads = cfg["num_v_heads"]
    num_k_heads = cfg["num_k_heads"]
    head_k_dim = cfg["head_k_dim"]
    head_v_dim = cfg["head_v_dim"]
    key_dim = cfg["key_dim"]
    value_dim = cfg["value_dim"]

    print(f"capture: layer_config dim={key_dim*2+value_dim} W={p['conv_weight'].shape[-1]}")
    print(f"n_actual={n_actual} n_acc={p['num_accepted_tokens'].tolist()} "
          f"spec_indices={p['spec_state_indices_tensor'].tolist()}")

    # ------------------------------------------------------------------
    # (A) Compare my torch conv1d output to FLA's causal_conv1d_update.
    # ------------------------------------------------------------------
    device = torch.device("xpu")
    mixed_qkv = p["mixed_qkv"][:n_actual].to(device)
    conv_w = p["conv_weight"]
    if conv_w.dim() == 3:
        conv_w = conv_w.view(conv_w.size(0), conv_w.size(2))
    conv_w = conv_w.to(device)
    conv_b = None if p["conv_bias"] is None else p["conv_bias"].to(device)
    activation = cfg["activation"]
    W = conv_w.shape[-1]

    # ---- Build dense conv_state pool that mirrors test harness layout ----
    slots_captured = p["slot_indices"].to(torch.long)
    real_slots = sorted({int(s) for s in slots_captured.tolist()} - {0})
    pool_size = len(real_slots) + 1
    max_slot = int(slots_captured.max().item()) + 1
    remap = torch.zeros(max_slot, dtype=torch.long)
    for dense_idx, src in enumerate(real_slots, start=1):
        remap[src] = dense_idx

    conv_state_pre = p["conv_state_pre"]                 # [n_slots, state_len, dim]
    state_len = conv_state_pre.shape[1]
    dim = conv_state_pre.shape[-1]
    conv_pool = torch.zeros((pool_size, state_len, dim),
                            dtype=conv_state_pre.dtype, device=device)
    slots_dense = remap[slots_captured]
    keep = slots_dense > 0
    conv_pool[slots_dense[keep]] = conv_state_pre[keep].to(device)

    # FLA's causal_conv1d_update expects (num_cache, dim, state_len).
    # Our pool layout is (num_cache, state_len, dim) — transpose for FLA call.
    conv_pool_for_fla = conv_pool.transpose(-1, -2).contiguous()  # [pool, dim, state_len]

    # FLA call setup matching _forward_core's spec branch.
    spec_idx_tensor = p["spec_state_indices_tensor"].to(device).to(torch.int32)
    num_acc = p["num_accepted_tokens"].to(device).to(torch.int32)
    spec_qsl = p["spec_query_start_loc"].to(device).to(torch.int32)

    # remap to dense
    spec_idx_dense = remap.to(device)[spec_idx_tensor.to(torch.long)].to(torch.int32)

    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_update,
    )

    mixed_qkv_for_fla = mixed_qkv.clone()  # FLA writes output back into x
    mixed_qkv_post_fla = causal_conv1d_update(
        mixed_qkv_for_fla,
        conv_pool_for_fla,
        conv_w,
        conv_b,
        activation,
        conv_state_indices=spec_idx_dense[:, 0][:int(p["num_spec_decodes"])],
        num_accepted_tokens=num_acc,
        query_start_loc=spec_qsl,
        max_query_len=spec_idx_tensor.size(-1),
        validate_data=False,
    )

    # My torch conv1d on the same captured pre-state.
    # CONV uses spec_state_indices[seq, 0] (the FIRST spec slot), NOT
    # [seq, n_acc-1] like ssm_state does. Asymmetry between conv & ssm slot
    # selection is the real point of confusion here.
    raw_load = int(p["spec_state_indices_tensor"][0, 0].item())
    state_idx_capture = (slots_captured == raw_load).nonzero(as_tuple=True)[0].item()
    conv_state_for_seq = conv_state_pre[state_idx_capture].to(device)  # [state_len, dim]
    conv_state_load = conv_state_for_seq[-(W - 1):]                     # [W-1, dim]
    mixed_qkv_post_torch = torch_causal_conv1d(
        mixed_qkv, conv_w, conv_b, conv_state_load, activation
    )

    delta_conv = (mixed_qkv_post_torch.float() - mixed_qkv_post_fla.float()).abs()
    print(f"\n=== (A) my torch conv1d  vs  FLA causal_conv1d_update ===")
    print(f"shape: {tuple(mixed_qkv_post_torch.shape)}")
    print(f"max abs diff: {delta_conv.max().item():.4e}  "
          f"mean: {delta_conv.mean().item():.4e}")
    tol_conv = 5e-3 + 5e-3 * mixed_qkv_post_fla.float().abs()
    fail_conv = delta_conv > tol_conv
    print(f"fail (atol=5e-3 + 5e-3*|fla|): {int(fail_conv.sum())}/"
          f"{fail_conv.numel()} ({100*fail_conv.float().mean():.4f}%)")
    # Per-token / per-channel structure
    per_token = fail_conv.float().sum(dim=1)
    print(f"per-token fails: {[int(x) for x in per_token.tolist()]}")

    # ------------------------------------------------------------------
    # (B) FLA recurrence on my torch-conv outputs vs FLA-captured outputs.
    # ------------------------------------------------------------------
    qkv_size = key_dim * 2 + value_dim
    assert mixed_qkv_post_torch.shape[1] == qkv_size

    # Use FLA's own conv output for (B) so we isolate the recurrence from any
    # remaining conv difference.
    mqp = mixed_qkv_post_fla
    q_torch = mqp[:, :key_dim].view(n_actual, num_k_heads, head_k_dim)
    k_torch = mqp[:, key_dim:2*key_dim].view(n_actual, num_k_heads, head_k_dim)
    v_torch = mqp[:, 2*key_dim:].view(n_actual, num_v_heads, head_v_dim)

    # Build dense ssm_state pool
    ssm_pre = p["ssm_state_pre"]
    ssm_pool = torch.zeros((pool_size,) + tuple(ssm_pre.shape[1:]),
                           dtype=ssm_pre.dtype, device=device)
    ssm_pool[slots_dense[keep]] = ssm_pre[keep].to(device)

    from vllm.model_executor.layers.fla.ops import (
        fused_sigmoid_gating_delta_rule_update,
    )
    # FLA expects q/k/v with leading B dim; pass batch=1
    q_b = q_torch.unsqueeze(0).contiguous()
    k_b = k_torch.unsqueeze(0).contiguous()
    v_b = v_torch.unsqueeze(0).contiguous()
    b_b = p["b"][:n_actual].to(device).unsqueeze(0).contiguous()
    a_b = p["a"][:n_actual].to(device).unsqueeze(0).contiguous()

    spec_qsl_for_fla = spec_qsl[: int(p["num_spec_decodes"]) + 1].contiguous()

    ssm_pool_for_fla = ssm_pool.clone()
    core_fla, _last = fused_sigmoid_gating_delta_rule_update(
        A_log=p["A_log"].to(device),
        a=a_b,
        b=b_b,
        dt_bias=p["dt_bias"].to(device),
        q=q_b,
        k=k_b,
        v=v_b,
        initial_state=ssm_pool_for_fla,
        inplace_final_state=True,
        cu_seqlens=spec_qsl_for_fla,
        ssm_state_indices=spec_idx_dense,
        num_accepted_tokens=num_acc,
        use_qk_l2norm_in_kernel=True,
    )
    core_fla_cpu = core_fla[0, :n_actual].float().cpu()

    fla_core_cap = p["core_attn_out"][:n_actual].float()
    delta_core = (core_fla_cpu - fla_core_cap).abs()
    tol_core = 2e-2 + 2e-2 * fla_core_cap.abs()
    fail_core = delta_core > tol_core
    print(f"\n=== (B) FLA-recurrence on my torch q/k/v  vs  FLA-captured core_attn_out ===")
    print(f"shape: {tuple(core_fla_cpu.shape)}")
    print(f"max abs diff: {delta_core.max().item():.4e}  "
          f"mean: {delta_core.mean().item():.4e}")
    print(f"fail (atol=2e-2 + 2e-2*|fla|): {int(fail_core.sum())}/"
          f"{fail_core.numel()} ({100*fail_core.float().mean():.4f}%)")

    # ssm_state_post diff
    pyref_ssm = ssm_pool_for_fla[slots_dense[keep]].cpu()  # in-place updated
    fla_ssm_cap = p["ssm_state_post"][keep.cpu()]
    delta_ssm = (pyref_ssm - fla_ssm_cap).abs()
    tol_ssm = 2e-2 + 2e-2 * fla_ssm_cap.abs()
    fail_ssm = delta_ssm > tol_ssm
    print(f"\n=== (B) FLA-recurrence ssm_state_post  vs  FLA-captured ssm_state_post ===")
    print(f"shape: {tuple(pyref_ssm.shape)}")
    print(f"max abs diff: {delta_ssm.max().item():.4e}  "
          f"mean: {delta_ssm.mean().item():.4e}")
    print(f"fail: {int(fail_ssm.sum())}/{fail_ssm.numel()} "
          f"({100*fail_ssm.float().mean():.4f}%)")

    print("\n=== verdict ===")
    a_pass = not fail_conv.any()
    b_pass = not fail_core.any() and not fail_ssm.any()
    print(f"(A) torch conv1d ≈ FLA causal_conv1d_update: {a_pass}")
    print(f"(B) FLA-recurrence on torch-conv outputs ≈ FLA-captured: {b_pass}")
    if a_pass and b_pass:
        print("→ My conv1d AND FLA pipeline agree with FLA-captured. ")
        print("  Pyref-of-hpp produces wrong output despite identical inputs and (claimed) identical math.")
        print("  Search for a transcription error in my pyref's gated_delta_rule transcription.")
    elif a_pass and not b_pass:
        print("→ Conv1d agrees, but FLA-recurrence on those q/k/v also fails. Strange — re-investigate.")
    elif not a_pass and not b_pass:
        print("→ Conv1d differs. The whole bug chain may stem from conv1d divergence.")
        print("  Re-run pyref-of-hpp with FLA-conv1d outputs as input to isolate gated_delta_rule.")
    else:
        print("→ Conv1d differs but FLA-recurrence on torch-conv matches FLA-captured. Suspicious.")


if __name__ == "__main__":
    sys.exit(main())

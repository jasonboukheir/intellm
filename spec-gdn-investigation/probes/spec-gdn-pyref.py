"""Pure-Python reference of `gated_delta_rule.hpp`'s compute, run on the
captured rung-4 inputs, diffed against FLA's captured outputs.

Goal: triangulate where the SYCL bug lives.
- pyref = transcription of gated_delta_rule.hpp's formulas in pure fp32 PyTorch.
- We synthesize post-conv q/k/v from `mixed_qkv` via a Python causal_conv1d,
  then feed q/k/v plus captured b/a/A_log/dt_bias/ssm_state_pre into pyref.
- Compare pyref output to FLA's captured ssm_state_post and core_attn_out.

Outcomes:
  pyref ≈ FLA captured  → algorithm-as-coded is correct, SYCL codegen bug.
  pyref ≠ FLA captured  → algorithm in gated_delta_rule.hpp diverges from FLA
                          at the math level (or my conv1d differs significantly
                          enough to mask the comparison).

Run inside the vllm-dev container:
  /opt/venv/bin/python /workspace/vllm/.spec-gdn-pyref.py
"""
from __future__ import annotations

import math
import sys

import torch

CAPTURE = (
    "/tmp/spec_gdn_captures/"
    "tuple_language_model_model_layers_0_linear_attn_000004_spec_K4_min4_max4.pt"
)


def causal_conv1d_python(
    x: torch.Tensor,           # [T, C] bf16  (mixed_qkv after rearrange)
    weight: torch.Tensor,      # [C, W] bf16  (conv1d weight, depthwise)
    bias: torch.Tensor | None, # [C] or None
    conv_state: torch.Tensor,  # [W-1, C]     prior W-1 tokens (newest last)
    activation: str,           # "silu" or "swish"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal depthwise conv1d. Replicates the standard mamba contract:
    out[t, c] = bias[c] + sum_{i=0..W-1} weight[c, i] * input[c, t-W+1+i]
    where input is conv_state ++ x along time axis.
    Activation applied at output (silu == swish).
    Returns (out [T, C], new_conv_state [W-1, C]).
    """
    T, C = x.shape
    W = weight.shape[-1]
    # Concatenate: full sequence of length conv_state_len + T
    full = torch.cat([conv_state.to(x.dtype), x], dim=0)  # [W-1+T, C]
    # Depthwise conv: per-channel sliding window
    out = torch.zeros((T, C), dtype=torch.float32, device=x.device)
    w = weight.float()  # [C, W]
    for t in range(T):
        # window covers full[t : t+W]   (since full has W-1 leading state)
        win = full[t:t + W].float()  # [W, C]
        out[t] = (win * w.t()).sum(dim=0)
    if bias is not None:
        out += bias.float().unsqueeze(0)
    if activation in ("silu", "swish"):
        out = out * torch.sigmoid(out)
    else:
        raise ValueError(f"unhandled activation: {activation}")
    new_state = full[T:T + (W - 1)] if T >= W - 1 else torch.cat(
        [conv_state[T:], full[-min(T, W - 1):]], dim=0
    )
    return out.to(x.dtype), new_state


def softplus(x: torch.Tensor, beta: float = 1.0, threshold: float = 20.0) -> torch.Tensor:
    """Match the kernel's act_softplus exactly."""
    bx = beta * x
    return torch.where(bx < threshold, torch.log1p(torch.exp(bx)) / beta, x)


def pyref_hpp(
    q: torch.Tensor,            # [T, num_k_heads, head_k_dim] bf16
    k: torch.Tensor,            # [T, num_k_heads, head_k_dim] bf16
    v: torch.Tensor,            # [T, num_v_heads, head_v_dim] bf16
    b: torch.Tensor,            # [T, num_v_heads] bf16
    a: torch.Tensor,            # [T, num_v_heads] bf16
    A_log: torch.Tensor,        # [num_v_heads] fp32
    dt_bias: torch.Tensor,      # [num_v_heads] bf16
    ssm_state_pool: torch.Tensor,  # [pool_size, num_v_heads, head_v_dim, head_k_dim] fp32
    qsl: torch.Tensor,          # [batch+1] cumulative seq lens
    spec_state_indices: torch.Tensor | None,  # [batch, K] or None
    num_accepted_tokens: torch.Tensor | None,  # [batch] or None
    cache_indices: torch.Tensor | None,        # [batch] or None (non-spec)
    has_initial_state: torch.Tensor | None,    # [batch] bool or None
    is_spec: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Faithful transcription of gated_delta_rule.hpp's compute. fp32 throughout.
    Per-(batch, head) block compute; the per-sub-group lane decomposition collapses
    to whole-head matmul/dot when expressed in PyTorch — but the math is identical
    (associative commutative reductions).

    Returns (core_attn_out [T, num_v_heads, head_v_dim] in same dtype as v,
             new_ssm_state_pool [pool_size, ...] fp32 — copy of input with writes).
    """
    T_total, num_v_heads, head_v_dim = v.shape
    num_k_heads = k.shape[1]
    head_k_dim = k.shape[2]
    kv_ratio = num_v_heads // num_k_heads
    eps = 1e-6
    scale = 1.0 / math.sqrt(head_k_dim)

    out_pool = ssm_state_pool.clone()
    core_attn_out = torch.zeros(
        T_total, num_v_heads, head_v_dim, dtype=v.dtype, device=v.device
    )

    A_log_neg_exp = -torch.exp(A_log)  # [num_v_heads]
    dt_bias_f = dt_bias.float()        # [num_v_heads]

    batch_size = qsl.numel() - 1
    for batch in range(batch_size):
        seq_start = int(qsl[batch].item())
        seq_end = int(qsl[batch + 1].item())
        if seq_end <= seq_start:
            continue

        # Resolve load_slot per the kernel's IS_SPEC vs !IS_SPEC branches.
        if is_spec:
            n_acc = int(num_accepted_tokens[batch].item())
            if n_acc <= 0:
                load_slot = 0
                has_init = False
            else:
                raw = int(spec_state_indices[batch, n_acc - 1].item())
                load_slot = raw
                has_init = raw > 0
        else:
            load_slot = int(cache_indices[batch].item())
            has_init = (
                has_initial_state is None or bool(has_initial_state[batch].item())
            )

        # Load initial state. Layout matches kernel index pattern (head, v, k).
        if has_init:
            state = out_pool[load_slot].clone().float()  # [num_v_heads, head_v_dim, head_k_dim]
        else:
            state = torch.zeros(
                num_v_heads, head_v_dim, head_k_dim,
                dtype=torch.float32, device=v.device,
            )

        for t in range(seq_start, seq_end):
            # Activations (per v_head)
            b_t = b[t].float()                       # [num_v_heads]
            beta = torch.sigmoid(b_t)                # [num_v_heads]
            a_t = a[t].float() + dt_bias_f           # [num_v_heads]
            g = torch.exp(A_log_neg_exp * softplus(a_t))  # [num_v_heads]

            # q, k for this token. Each v_head h reads from k_head = h // kv_ratio.
            q_t = q[t].float()  # [num_k_heads, head_k_dim]
            k_t = k[t].float()  # [num_k_heads, head_k_dim]

            # L2 normalize q and k inside the kernel (matches FLA USE_QK_L2NORM_IN_KERNEL).
            q_sum_per_kh = (q_t * q_t).sum(dim=-1) + eps  # [num_k_heads]
            k_sum_per_kh = (k_t * k_t).sum(dim=-1) + eps  # [num_k_heads]
            q_norm_per_kh = q_t / torch.sqrt(q_sum_per_kh).unsqueeze(-1)
            q_norm_per_kh = q_norm_per_kh * scale
            k_norm_per_kh = k_t / torch.sqrt(k_sum_per_kh).unsqueeze(-1)

            # Broadcast q/k from k-heads to v-heads (GQA)
            kh_per_vh = torch.arange(num_v_heads) // kv_ratio  # [num_v_heads]
            q_norm = q_norm_per_kh[kh_per_vh]  # [num_v_heads, head_k_dim]
            k_norm = k_norm_per_kh[kh_per_vh]  # [num_v_heads, head_k_dim]

            v_t = v[t].float()  # [num_v_heads, head_v_dim]

            # state[h, v, k] *= g[h]
            state = state * g.view(num_v_heads, 1, 1)

            # kv_mem[h, v] = sum_k state[h, v, k] * k_norm[h, k]
            kv_mem = torch.einsum("hvk,hk->hv", state, k_norm)  # [num_v_heads, head_v_dim]

            # delta[h, v] = (v_t[h, v] - kv_mem[h, v]) * beta[h]
            delta = (v_t - kv_mem) * beta.unsqueeze(-1)         # [num_v_heads, head_v_dim]

            # state[h, v, k] += k_norm[h, k] * delta[h, v]
            state = state + delta.unsqueeze(-1) * k_norm.unsqueeze(1)

            # res[h, v] = sum_k state[h, v, k] * q_norm[h, k]
            res = torch.einsum("hvk,hk->hv", state, q_norm)     # [num_v_heads, head_v_dim]

            core_attn_out[t] = res.to(core_attn_out.dtype)

            if is_spec:
                spec_slot = int(spec_state_indices[batch, t - seq_start].item())
                if spec_slot > 0:
                    out_pool[spec_slot] = state.to(out_pool.dtype)

        if not is_spec:
            out_pool[load_slot] = state.to(out_pool.dtype)

    return core_attn_out, out_pool


def main():
    p = torch.load(CAPTURE, map_location="cpu", weights_only=False)
    cfg = p["layer_config"]
    num_v_heads = cfg["num_v_heads"]
    num_k_heads = cfg["num_k_heads"]
    head_k_dim = cfg["head_k_dim"]
    head_v_dim = cfg["head_v_dim"]
    key_dim = cfg["key_dim"]
    value_dim = cfg["value_dim"]
    activation = cfg["activation"]

    n_actual = int(p["num_actual_tokens"])
    print(f"capture: {CAPTURE}")
    print(f"layer_config: V={num_v_heads} K={num_k_heads} HK={head_k_dim} "
          f"HV={head_v_dim} key_dim={key_dim} value_dim={value_dim} act={activation}")
    print(f"n_actual={n_actual} num_spec_decodes={int(p['num_spec_decodes'])} "
          f"num_accepted_tokens={p['num_accepted_tokens'].tolist()}")

    # ---- Reconstruct dense pool (mirroring _build_dense_pool) ----
    slots_captured = p["slot_indices"].to(torch.long)
    real_slots = sorted({int(s) for s in slots_captured.tolist()} - {0})
    pool_size = len(real_slots) + 1
    max_slot = int(slots_captured.max().item()) + 1
    remap = torch.zeros(max_slot, dtype=torch.long)
    for dense_idx, src_slot in enumerate(real_slots, start=1):
        remap[src_slot] = dense_idx

    ssm_pre = p["ssm_state_pre"]  # [n_slots, V, HV, HK] fp32
    ssm_pool = torch.zeros(
        (pool_size,) + tuple(ssm_pre.shape[1:]), dtype=ssm_pre.dtype
    )
    slots_dense = remap[slots_captured]
    keep = slots_dense > 0
    ssm_pool[slots_dense[keep]] = ssm_pre[keep]

    # ---- Reconstruct post-conv q/k/v from mixed_qkv ----
    mixed_qkv = p["mixed_qkv"][:n_actual]   # [T, qkv_size]
    conv_w = p["conv_weight"]               # [C, 1, W] or [C, W]
    if conv_w.dim() == 3:
        conv_w = conv_w.view(conv_w.size(0), conv_w.size(2))   # [C, W]
    conv_b = p["conv_bias"]
    conv_state_pre_full = p["conv_state_pre"]   # [n_slots, conv_state_len, C]

    qkv_size = key_dim * 2 + value_dim
    assert mixed_qkv.shape[1] == qkv_size, (mixed_qkv.shape, qkv_size)

    # CONV's load slot is spec_state_indices[seq, 0] (first spec slot), NOT
    # [seq, n_acc-1] like ssm_state. The conv state at slot[0] holds the
    # rolled history; spec_state_indices_tensor[0, 0] is the conv anchor.
    raw_load_slot = int(p["spec_state_indices_tensor"][0, 0].item())
    conv_state_idx = (slots_captured == raw_load_slot).nonzero(as_tuple=True)[0].item()
    conv_state_load = conv_state_pre_full[conv_state_idx]   # [conv_state_len, C]
    W = conv_w.shape[-1]
    conv_state_load = conv_state_load[-(W - 1):]   # take last W-1 rows
    print(f"conv_state load_slot={raw_load_slot} state_idx={conv_state_idx} "
          f"W={W} conv_state_load.shape={tuple(conv_state_load.shape)}")

    mixed_qkv_post, _ = causal_conv1d_python(
        mixed_qkv, conv_w, conv_b, conv_state_load, activation
    )

    q_post = mixed_qkv_post[:, :key_dim].view(n_actual, num_k_heads, head_k_dim)
    k_post = mixed_qkv_post[:, key_dim:2 * key_dim].view(n_actual, num_k_heads, head_k_dim)
    v_post = mixed_qkv_post[:, 2 * key_dim:].view(n_actual, num_v_heads, head_v_dim)

    # ---- Build qsl/spec/idx tensors (dense slot space) ----
    qsl = torch.tensor([0, n_actual], dtype=torch.long)
    spec_idx_dense = remap[p["spec_state_indices_tensor"].to(torch.long)]
    num_acc = p["num_accepted_tokens"].to(torch.long)

    # ---- Run pyref ----
    print("\nRunning pyref_hpp...")
    pyref_core_out, pyref_pool = pyref_hpp(
        q_post, k_post, v_post,
        p["b"][:n_actual], p["a"][:n_actual],
        p["A_log"], p["dt_bias"],
        ssm_pool,
        qsl,
        spec_idx_dense,
        num_acc,
        cache_indices=None,
        has_initial_state=None,
        is_spec=True,
    )

    # ---- Diff against captured FLA outputs ----
    fla_core = p["core_attn_out"][:n_actual].float()
    pyref_core_f = pyref_core_out.float()
    delta_core = (pyref_core_f - fla_core).abs()
    tol_core = 2e-2 + 2e-2 * fla_core.abs()
    fail_core = delta_core > tol_core
    print(f"\n=== pyref_hpp.core_attn_out vs FLA captured ===")
    print(f"shape: {tuple(pyref_core_f.shape)}")
    print(f"max abs diff: {delta_core.max().item():.4e}  "
          f"mean: {delta_core.mean().item():.4e}")
    print(f"fail (atol=2e-2 + 2e-2*|fla|): "
          f"{int(fail_core.sum())}/{fail_core.numel()} "
          f"({100*fail_core.float().mean():.4f}%)")
    if fail_core.any():
        # Per-token / per-(head, v) breakdown
        per_tok = fail_core.float().sum(dim=(1, 2))
        print(f"per-token fails: {[int(x) for x in per_tok.tolist()]}")
        worst = (delta_core * fail_core.float()).flatten().topk(10).indices
        for idx in worst:
            t, h, vv = (
                int(idx) // (num_v_heads * head_v_dim),
                (int(idx) // head_v_dim) % num_v_heads,
                int(idx) % head_v_dim,
            )
            print(f"  (t={t}, h={h}, v={vv}): "
                  f"pyref={pyref_core_f[t, h, vv].item():+.4e} "
                  f"fla={fla_core[t, h, vv].item():+.4e} "
                  f"delta={delta_core[t, h, vv].item():.4e}")

    # ssm_state_post
    fla_ssm = p["ssm_state_post"]   # [n_slots, V, HV, HK] fp32
    pyref_ssm = pyref_pool[slots_dense[keep]]   # [n_slots, V, HV, HK]
    fla_ssm_keep = fla_ssm[keep]
    delta_ssm = (pyref_ssm - fla_ssm_keep).abs()
    tol_ssm = 2e-2 + 2e-2 * fla_ssm_keep.abs()
    fail_ssm = delta_ssm > tol_ssm
    print(f"\n=== pyref_hpp.ssm_state_post vs FLA captured ===")
    print(f"shape: {tuple(pyref_ssm.shape)}  slots: {real_slots}")
    print(f"max abs diff: {delta_ssm.max().item():.4e}  "
          f"mean: {delta_ssm.mean().item():.4e}")
    print(f"fail (atol=2e-2 + 2e-2*|fla|): "
          f"{int(fail_ssm.sum())}/{fail_ssm.numel()} "
          f"({100*fail_ssm.float().mean():.4f}%)")
    if fail_ssm.any():
        per_slot = fail_ssm.float().sum(dim=(1, 2, 3))
        print(f"per-slot fails: {[int(x) for x in per_slot.tolist()]}")
        for s_idx in range(pyref_ssm.shape[0]):
            n_fail = int(fail_ssm[s_idx].sum())
            if n_fail == 0:
                continue
            worst_per_slot = (delta_ssm[s_idx] * fail_ssm[s_idx].float()).flatten().topk(5).indices
            print(f"  slot dense={int(slots_dense[keep][s_idx])} "
                  f"src={int(slots_captured[keep][s_idx])} fails={n_fail}:")
            for idx in worst_per_slot:
                tup = torch.unravel_index(idx, delta_ssm[s_idx].shape)
                h, vv, kk = (int(x) for x in tup)
                print(f"    (h={h}, v={vv}, k={kk}): "
                      f"pyref={pyref_ssm[s_idx, h, vv, kk].item():+.4e} "
                      f"fla={fla_ssm_keep[s_idx, h, vv, kk].item():+.4e} "
                      f"delta={delta_ssm[s_idx, h, vv, kk].item():.4e}")

    print("\n=== verdict ===")
    if not fail_core.any() and not fail_ssm.any():
        print("pyref_hpp ≈ FLA captured at atol/rtol=2e-2.")
        print("Implication: gated_delta_rule.hpp's formulas match FLA mathematically;")
        print("the SYCL-vs-FLA divergence must come from compiled-kernel codegen,")
        print("not from the algorithm as written. Next experiment: SPIR-V / IR inspection.")
    else:
        print("pyref_hpp ≠ FLA captured.")
        print("Implication: either (a) gated_delta_rule.hpp's formulas diverge from FLA")
        print("at the math level, or (b) my conv1d/transcription is off enough to mask")
        print("the comparison. Need to isolate which.")


if __name__ == "__main__":
    sys.exit(main())

"""Tick-25 reframe: cheap probes before any rebuild.

Runs three independent checks on the layer-0 worst capture, all at
g=1 (A_log=-100) and beta=0 (b=-100):

  (P1) HOST PYREF kv_mem & delta finite-check at j=3 stripe.
       Computes kv_mem and delta in fp32 on the host from captured
       state_pre and conv-output k. If kv_mem[j=3] is ever NaN/Inf or
       |kv_mem| > 1e30, then `delta = (v - kv_mem) * 0` could become
       NaN under non-IEEE flushing in IGC AOT, masquerading as a
       codegen bug. If all kv_mem values are finite and reasonable,
       the codegen-bug framing stands.

  (P2) IS_SPEC=false LOAD-READBACK using the SYCL kernel.
       Same g=1, beta=0 setup, but routes through the dispatcher's
       non-spec NATIVE_LAUNCHER path so the IS_SPEC=false template
       runs the multi-token chunk loop. Final-state writeback is to
       a single slot per batch. Diff post vs pre.
       - Drift > bf16 noise → bug is universal, not template-specific
         (rules out the IS_SPEC=true RA hypothesis from T23).
       - Drift = 0 → bug IS spec-template-specific; the difference
         between IS_SPEC=true and IS_SPEC=false RA actually matters.

We can't compute `k_local` purely host-side (it's the post-conv1d v
projection, computed on XPU); we use the captured `q`/`k`/`v` post-
conv tensors when the capture provides them, otherwise re-derive
post-conv k by running the kernel with sentinel inputs. For tick 25
we only need the host kv_mem ROUGH magnitude check — using captured
state_pre and any reasonable k magnitude proxy is fine.
"""

from __future__ import annotations

import math
import sys

import torch

import vllm._xpu_ops  # noqa: F401
import vllm_xpu_kernels._xpu_C  # noqa: F401

sys.path.insert(0, "/workspace/vllm")

from tests.kernels.xpu.test_spec_gdn_replay import (  # noqa: E402
    _build_dense_pool,
    _remap_index_tensor,
)

CAPTURE = (
    "/tmp/spec_gdn_captures/"
    "tuple_language_model_model_layers_0_linear_attn_000004_spec_K4_min4_max4.pt"
)


def p1_pyref_kv_mem(payload):
    """Host-side worst-case check: with g=1 the LOADED state_local *should*
    satisfy `kv_mem[j] = sum_i state_local[j*K+i] * k_local[i]` via a
    sub-group reduction. Because kv_mem reduction sums across the 32
    sg-lanes (each lane holds k_bucket_size=4 (i, j) chunks for one
    head_v_dim_id stripe), the reduction sums `head_k_dim` (=128)
    products. We can bound |kv_mem| above by max|state| * max|k| *
    head_k_dim and check that against fp32 overflow (~3.4e38).
    """
    state_pre = payload["ssm_state_pre"].float()  # (slot, head, V, K)
    # k post-conv isn't in the capture, but conv-projection magnitudes
    # are bounded by conv_weight * qkvz magnitudes plus bias.
    qkvz = payload["projected_states_qkvz"].float()
    conv_w = payload["conv_weight"].float()
    if conv_w.dim() == 3:
        conv_w = conv_w.view(conv_w.size(0), conv_w.size(2))
    # crude upper bound: |k_post_conv| <= |conv_w| * |qkvz| * width
    width = conv_w.size(1)
    k_max = (qkvz.abs().max().item()
             * conv_w.abs().max().item()
             * width)
    s_max = state_pre.abs().max().item()
    head_k_dim = state_pre.size(-1)  # 128
    kv_mem_bound = s_max * k_max * head_k_dim

    print(f"[P1] state_pre  max abs = {s_max:.4e}")
    print(f"[P1] qkvz       max abs = {qkvz.abs().max().item():.4e}")
    print(f"[P1] conv_w     max abs = {conv_w.abs().max().item():.4e}  "
          f"(width={width})")
    print(f"[P1] |k_post_conv| upper bound = {k_max:.4e}")
    print(f"[P1] |kv_mem|      upper bound = {kv_mem_bound:.4e}")
    print(f"[P1] fp32 overflow at ~3.4e38; safety margin = "
          f"{(3.4e38 / max(kv_mem_bound, 1e-30)):.4e}x")

    # Sanity-check beta=0 in pyref too: `beta = sigmoid(-100)`.
    beta = 1.0 / (1.0 + math.exp(100.0))
    print(f"[P1] beta = sigmoid(-100) = {beta:.4e} (denormal-flush → 0)")
    delta_bound = (s_max + k_max) * beta  # |v - kv_mem| <= |v| + |kv_mem|
    print(f"[P1] |delta|       upper bound = {delta_bound:.4e}")
    if not (kv_mem_bound < 1e30 and delta_bound < 1e-30):
        print("[P1] WARN: bound is large — investigate per-(h, j) detail.")
    else:
        print("[P1] PASS: kv_mem bounded well below overflow; delta bounded "
              "well below denormal threshold. delta=0 is a robust "
              "assumption in IEEE math.")


def p2_isspec_false_readback(payload):
    """Run NATIVE_LAUNCHER with IS_SPEC=false and the multi-token loop;
    diff post-state vs pre-state at the load slot."""
    payload = dict(payload)
    payload["A_log"] = torch.full_like(payload["A_log"], -100.0)
    payload["b"] = torch.full_like(payload["b"], -100.0)

    device = torch.device("xpu")
    n_actual = int(payload["num_actual_tokens"])
    cfg = payload["layer_config"]

    conv_pool, ssm_pool, remap, _ = _build_dense_pool(payload, device)
    ssm_pool_pre = ssm_pool.clone()

    # Reuse the spec-capture tensors but route through the
    # IS_SPEC=false dispatch: pass non-spec state-index per batch
    # (= the slot the spec capture loaded from = spec_state_indices[b, N-1])
    # and non-spec qsl that mirrors the spec qsl (4-token spans).
    spec_idx = payload["spec_state_indices_tensor"]
    num_acc = payload["num_accepted_tokens"].to(torch.long)
    spec_qsl = payload["spec_query_start_loc"].to(device).contiguous()

    # Build per-batch single-slot index from the spec ring's last accepted
    # token slot (which is what FLA uses for the IS_SPEC=false load slot).
    n_batches = num_acc.numel()
    nonspec_idx_host = torch.empty(n_batches, dtype=torch.long)
    for b in range(n_batches):
        n = int(num_acc[b].item())
        nonspec_idx_host[b] = spec_idx[b, n - 1] if n > 0 else 0
    # Remap to dense pool indices.
    nonspec_idx_dense = _remap_index_tensor(
        nonspec_idx_host.to(torch.int32), remap
    ).to(device).to(torch.int32).contiguous()

    qkvz = payload["projected_states_qkvz"].to(device)
    ba = payload["projected_states_ba"].to(device)
    core_attn_out = torch.empty(
        payload["core_attn_out"].shape,
        dtype=payload["core_attn_out"].dtype, device=device)
    z = torch.empty(payload["z"].shape, dtype=payload["z"].dtype,
                    device=device)
    conv_w = payload["conv_weight"].to(device)
    if conv_w.dim() == 3:
        conv_w = conv_w.view(conv_w.size(0), conv_w.size(2))
    conv_b = payload.get("conv_bias")
    if conv_b is not None:
        conv_b = conv_b.to(device)

    has_initial_state = payload.get("has_initial_state")
    if has_initial_state is not None:
        has_initial_state = has_initial_state.to(device)

    # IMPORTANT: do NOT pass spec_state_indices_tensor / num_accepted_tokens.
    # The dispatcher then sets is_spec=false → IS_SPEC=false template →
    # multi-token chunk loop with a single per-batch slot for load+store.
    torch.ops._xpu_C.gdn_attention(
        core_attn_out[:n_actual],
        z[:n_actual],
        qkvz[:n_actual].contiguous(),
        ba[:n_actual].contiguous(),
        cfg["num_k_heads"],
        cfg["num_v_heads"],
        cfg["head_k_dim"],
        cfg["head_v_dim"],
        num_prefills=0,
        num_decodes=n_batches,
        has_initial_state=has_initial_state,
        non_spec_query_start_loc=spec_qsl,
        non_spec_state_indices_tensor=nonspec_idx_dense,
        num_actual_tokens=n_actual,
        conv_state=conv_pool,
        ssm_state=ssm_pool,
        conv_weights=conv_w,
        conv_bias=conv_b,
        activation=cfg["activation"],
        A_log=payload["A_log"].to(device),
        dt_bias=payload["dt_bias"].to(device),
        tp_size=cfg["tp_size"],
        reorder_input=not cfg["gqa_interleaved_layout"],
    )

    pre = ssm_pool_pre.cpu().float()
    post = ssm_pool.cpu().float()
    print()
    print("[P2] IS_SPEC=false multi-token (4 iters) load-readback under g=1, beta=0")
    print("[P2] Expected: post[load_slot] == pre[load_slot] byte-equal.")

    overall_max = 0.0
    overall_hot = 0
    for b in range(n_batches):
        slot = int(nonspec_idx_dense[b].item())
        if slot <= 0:
            continue
        d = (post[slot] - pre[slot]).abs()
        m = d.max().item()
        hot = int((d > 1e-3).sum().item())
        per_v = d.amax(dim=(0, 2))  # (V,)
        top_v = torch.topk(per_v, 5)
        print(f"  batch {b} slot={slot}: max={m:.4e}  hot(>{1e-3:.0e})={hot}"
              f"  top-v: " +
              ", ".join(f"v={int(i)}({m_:.3f})"
                        for m_, i in zip(top_v.values.tolist(),
                                         top_v.indices.tolist())))
        overall_max = max(overall_max, m)
        overall_hot += hot

    print(f"[P2] cross-batch: max={overall_max:.4e}  total_hot={overall_hot}")
    if overall_max < 1e-3:
        print("[P2] CLEAN under IS_SPEC=false: bug is IS_SPEC=true specific.")
    else:
        print("[P2] DRIFT under IS_SPEC=false too: bug is universal in the "
              "NATIVE multi-token loop, not spec-template-specific.")


def main():
    payload = torch.load(CAPTURE, map_location="cpu", weights_only=False)
    print("=" * 72)
    print("Tick 25 reframe — cheap probes (no rebuild)")
    print("=" * 72)
    p1_pyref_kv_mem(payload)
    p2_isspec_false_readback(payload)


if __name__ == "__main__":
    sys.exit(main() or 0)

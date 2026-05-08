"""Tick-19 load-readback probe.

Tick 18 isolated the bug to LOAD (gated_delta_rule.hpp:144-149). This
script exercises only the load by making iter 0..3 no-ops:
- A_log = -100  ⇒  g = exp(-exp(-100) * softplus(...)) ≡ 1.0
- b     = -100  ⇒  beta = sigmoid(-100) ≡ 0.0  ⇒  delta = 0
Therefore state_local stays unchanged across all 4 iterations, so
the writeback at every spec slot = state_local = the LOADED value
(carrying any load-side defect).

Diff `ssm_state_post[spec_slot]` against the original
`ssm_state_pre[load_slot]` cell-by-cell. Disagreements name the load
defect: for each (h, v, k) cell where they differ, we know what the
SYCL kernel actually loaded vs. what the captured state contained.

Layout (per tick-4 axis correction): (slot, head, v, k).
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
    _remap_index_tensor,
)

CAPTURE = (
    "/tmp/spec_gdn_captures/"
    "tuple_language_model_model_layers_0_linear_attn_000004_spec_K4_min4_max4.pt"
)


def main():
    payload = torch.load(CAPTURE, map_location="cpu", weights_only=False)
    payload["__path__"] = CAPTURE
    n_actual = int(payload["num_actual_tokens"])
    device = torch.device("xpu")

    payload_n = dict(payload)
    A_log_orig = payload["A_log"]
    payload_n["A_log"] = torch.full_like(A_log_orig, -100.0)
    b_orig = payload["b"]
    payload_n["b"] = torch.full_like(b_orig, -100.0)

    conv_pool, ssm_pool, remap, _ = _build_dense_pool(payload_n, device)
    ssm_pool_orig = ssm_pool.clone()

    sycl_core, _, _, ssm_pool_post = _call_sycl(
        payload_n, conv_pool, ssm_pool, remap, device
    )

    spec_idx_dense = _remap_index_tensor(
        payload_n["spec_state_indices_tensor"], remap
    ).to(torch.long)
    num_acc = payload_n["num_accepted_tokens"].to(torch.long).cpu()
    n_batches = num_acc.numel()

    print("=" * 72)
    print(f"Load-readback probe — {n_batches} batches, K=4, all iters no-op")
    print("=" * 72)

    pre_cpu = ssm_pool_orig.cpu().float()
    post_cpu = ssm_pool_post.cpu().float()

    all_diff_count = 0
    all_max = 0.0
    head_v_fail_count = torch.zeros(32, 128, dtype=torch.int64)

    for b in range(n_batches):
        n = int(num_acc[b].item())
        if n <= 0:
            continue
        load_slot = int(spec_idx_dense[b, n - 1].item())
        if load_slot <= 0:
            continue

        pre = pre_cpu[load_slot]  # (H, V, K)
        deltas = []
        for t in range(n):
            spec_slot = int(spec_idx_dense[b, t].item())
            if spec_slot <= 0:
                continue
            post = post_cpu[spec_slot]  # (H, V, K)
            delta = (post - pre).abs()
            deltas.append((t, spec_slot, delta))

        if not deltas:
            continue
        # All iters should be byte-equal (no-op chunk loop).
        # Pairwise iter-vs-iter disagreement maps drift growth.
        print(f"  batch {b}: load_slot={load_slot} (= iter-3 write slot)")
        slots = [post_cpu[s] for (_, s, _) in deltas]
        for i in range(len(slots)):
            for j in range(i + 1, len(slots)):
                d_ij = (slots[i] - slots[j]).abs()
                # Per-(head, v) max
                d_hv = d_ij.amax(dim=2)  # (H, V)
                hot = (d_hv > 1e-3).sum().item()
                print(
                    f"    iter{i} vs iter{j}: max={d_ij.max().item():.4e} "
                    f"mean={d_ij.mean().item():.4e} "
                    f"hot(h,v)={hot}"
                )
        iter0_delta = deltas[0][2]

        d = iter0_delta
        all_diff_count += int((d > 1e-3).sum().item())
        all_max = max(all_max, d.max().item())

        # Per-(head, v) max (over k) for this batch
        per_hv_max = d.amax(dim=2)  # (H, V)
        head_v_fail_count += (per_hv_max > 1e-3).to(torch.int64)

        if b == 0:
            print(f"  batch 0 detail: shape {tuple(d.shape)}, "
                  f"max {d.max().item():.4f}, mean {d.mean().item():.4e}")
            top_k_per_v = d.amax(dim=(0,))  # (V, K) → max over heads
            top_v_max = top_k_per_v.amax(dim=1)  # (V,)
            top_v = torch.topk(top_v_max, 8)
            print("  batch 0 top-8 v_dim by max load defect: ", end="")
            print(", ".join(f"v={int(i)}({m:.3f})"
                            for m, i in zip(top_v.values.tolist(),
                                            top_v.indices.tolist())))

            # focus on (h=14, v=79)
            print("  batch 0 (h=14, v=79) k-cells with |delta| > 1e-3:")
            for k in range(d.shape[2]):
                v = d[14, 79, k].item()
                if v > 1e-3:
                    print(
                        f"    k={k:>3}: "
                        f"pre={pre[14, 79, k].item():>9.4f} "
                        f"post={post_cpu[deltas[0][1]][14, 79, k].item():>9.4f} "
                        f"delta={v:>9.4f}"
                    )

    print()
    print(f"Cross-batch summary: total cells with |delta|>1e-3 = "
          f"{all_diff_count}, max |delta| = {all_max:.4f}")
    print()

    # Aggregate (h, v) where load defect appears
    per_v = head_v_fail_count.sum(dim=0)  # over heads
    per_h = head_v_fail_count.sum(dim=1)  # over v
    top_v = torch.topk(per_v, 10)
    print("Top-10 v_dim by # of (head, batch) with load defect: ", end="")
    print(", ".join(f"v={int(i)}({int(c)})"
                    for c, i in zip(top_v.values.tolist(),
                                    top_v.indices.tolist())))
    top_h = torch.topk(per_h, 10)
    print("Top-10 head by # of (v, batch) with load defect: ", end="")
    print(", ".join(f"h={int(i)}({int(c)})"
                    for c, i in zip(top_h.values.tolist(),
                                    top_h.indices.tolist())))


if __name__ == "__main__":
    sys.exit(main() or 0)

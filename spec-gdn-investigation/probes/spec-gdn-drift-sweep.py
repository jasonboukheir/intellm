"""Tick-25 cross-capture drift sweep.

Reviewer pushback: "drift = 0" claim on layer-0 worst capture (tick 20-22)
might be a happy alignment for that input distribution. Real signal is
iter0-vs-iter3 drift = 0 across ALL 80 rung-4 (`spec_K4_min4_max4`)
captures.

Setup (same as tick 19): A_log = -100  →  g ≡ 1.0, b = -100  →  beta ≡ 0.
Mathematically iter t = identity, so the per-iter chunk-loop body should
be a no-op and state at every spec slot should be byte-equal.

Output: per-(layer, capture) max iter0-vs-iter3 |delta|, and an aggregate
verdict — universal drift=0 (diagnosis confirmed) vs partial (incomplete).

Run inside vllm-dev:
  vllm-run /opt/venv/bin/python /workspace/vllm/.spec-gdn-drift-sweep.py
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

import vllm._xpu_ops  # noqa: F401
import vllm_xpu_kernels._xpu_C  # noqa: F401

sys.path.insert(0, "/workspace/vllm")
from tests.kernels.xpu.test_spec_gdn_replay import (  # noqa: E402
    _build_dense_pool,
    _call_sycl,
    _remap_index_tensor,
)

CAPTURES = Path("/tmp/spec_gdn_captures")
LAYER_RE = re.compile(r"layers_(\d+)_linear_attn_\d+_spec_K4_min4_max4\.pt$")


def per_capture_drift(payload, device):
    payload_n = dict(payload)
    payload_n["A_log"] = torch.full_like(payload["A_log"], -100.0)
    payload_n["b"] = torch.full_like(payload["b"], -100.0)

    conv_pool, ssm_pool, remap, _ = _build_dense_pool(payload_n, device)
    if conv_pool is None:
        return None

    _, _, _, ssm_pool_post = _call_sycl(
        payload_n, conv_pool, ssm_pool, remap, device
    )

    spec_idx_dense = _remap_index_tensor(
        payload_n["spec_state_indices_tensor"], remap
    ).to(torch.long)
    num_acc = payload_n["num_accepted_tokens"].to(torch.long).cpu()
    n_batches = num_acc.numel()

    post_cpu = ssm_pool_post.cpu().float()

    drift_max = 0.0
    drift_hot = 0
    n_pairs = 0

    for b in range(n_batches):
        n = int(num_acc[b].item())
        if n <= 1:
            continue
        slots = []
        for t in range(n):
            slot = int(spec_idx_dense[b, t].item())
            if slot > 0:
                slots.append(post_cpu[slot])
        if len(slots) < 2:
            continue
        # Compare iter 0 (slot t=0) vs iter (n-1) (slot t=n-1).
        d = (slots[0] - slots[-1]).abs()
        drift_max = max(drift_max, d.max().item())
        drift_hot += int((d > 1e-3).sum().item())
        n_pairs += 1

    return dict(
        drift_max=drift_max,
        drift_hot=drift_hot,
        n_pairs=n_pairs,
    )


def main():
    device = torch.device("xpu")
    paths = sorted(CAPTURES.glob("tuple_*spec_K4_min4_max4.pt"))
    print(f"[drift-sweep] {len(paths)} rung-4 captures", flush=True)

    by_layer = defaultdict(list)
    universal_zero = True
    overall_max = 0.0
    overall_hot = 0
    nonzero_captures = 0

    for i, p in enumerate(paths):
        m = LAYER_RE.search(p.name)
        if not m:
            continue
        layer = int(m.group(1))
        payload = torch.load(p, map_location="cpu", weights_only=False)
        payload["__path__"] = str(p)
        try:
            s = per_capture_drift(payload, device)
        except Exception as e:
            print(f"[{i+1}/{len(paths)}] {p.name}: ERROR "
                  f"{type(e).__name__}: {e}", flush=True)
            continue
        if s is None:
            continue
        by_layer[layer].append(s)
        if s["drift_max"] > 1e-6:
            universal_zero = False
            nonzero_captures += 1
        overall_max = max(overall_max, s["drift_max"])
        overall_hot += s["drift_hot"]
        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(paths)}] running max={overall_max:.4e} "
                  f"hot={overall_hot} nonzero_caps={nonzero_captures}",
                  flush=True)

    print()
    print("=== per-layer aggregate (iter 0 vs iter n_acc-1, g=1, beta=0) ===")
    print(f"{'layer':>5} {'n':>3} {'max(over caps)':>16} "
          f"{'mean(max)':>12} {'hot_total':>10}")
    for layer in sorted(by_layer):
        stats = by_layer[layer]
        max_d = max(s["drift_max"] for s in stats)
        mean_d = sum(s["drift_max"] for s in stats) / len(stats)
        hot = sum(s["drift_hot"] for s in stats)
        print(f"{layer:>5} {len(stats):>3} {max_d:>16.4e} "
              f"{mean_d:>12.4e} {hot:>10}")

    print()
    print("=== verdict ===")
    print(f"captures_total      = {sum(len(v) for v in by_layer.values())}")
    print(f"captures_nonzero    = {nonzero_captures}")
    print(f"overall_max_drift   = {overall_max:.6e}")
    print(f"overall_hot_cells   = {overall_hot}")
    print(f"universal_drift_zero = {universal_zero}")
    if universal_zero:
        print()
        print("DIAGNOSIS CONFIRMED: with current build (self-assign at line "
              "242), iter0-vs-iter(n_acc-1) drift is 0 across all rung-4 "
              "captures. The compiler-elision narrative holds at scale.")
    else:
        print()
        print("DIAGNOSIS INCOMPLETE: some captures still drift even with "
              "self-assign. Other code paths perturb state_local across "
              "iterations.")


if __name__ == "__main__":
    sys.exit(main() or 0)

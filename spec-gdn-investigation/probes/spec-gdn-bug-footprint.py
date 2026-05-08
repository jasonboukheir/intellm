"""Bug-footprint aggregator: walks every rung-3 (spec_K4_min1_max1) and
rung-4 (spec_K4_min4_max4) capture, runs SYCL + inline FLA oracle on each,
and aggregates per-layer:
  (rung, layer_id) -> (max abs, mean abs, fail_count / total, top v_dim coords by fail count)

Run inside vllm-dev:
  vllm-run /opt/venv/bin/python /workspace/vllm/.spec-gdn-bug-footprint.py

Goal (tick 12): confirm tick 9's hypothesis that v_dim coords
[25,33,47,53,55,60,79,83,107,124] recur across layers and that layer 0
has the worst amplitude. A confirmed pattern gives a tight bisect target
for the SPIR-V/IR inspection.
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

import vllm._xpu_ops  # noqa: F401  (registers _gdn_xpu)
import vllm_xpu_kernels._xpu_C  # noqa: F401  (registers _xpu_C.gdn_attention)

sys.path.insert(0, "/workspace/vllm")
from tests.kernels.xpu.test_spec_gdn_replay import (
    _build_dense_pool,
    _call_sycl,
    _compute_fla_spec_oracle,
)

CAPTURES = Path("/tmp/spec_gdn_captures")
LAYER_RE = re.compile(r"layers_(\d+)_linear_attn_\d+_(spec_K4_min[14]_max[14])\.pt$")


def per_capture_stats(payload, device):
    n_actual = int(payload["num_actual_tokens"])
    conv_pool, ssm_pool, remap, _ = _build_dense_pool(payload, device)
    if conv_pool is None:
        return None

    sycl_core, _, _, sycl_ssm_post = _call_sycl(
        payload, conv_pool, ssm_pool, remap, device
    )
    oracle_core, _, oracle_ssm_post, _ = _compute_fla_spec_oracle(payload, device)
    if oracle_core is None:
        return None

    # core_attn_out diff: shape (num_tokens, num_v_heads, head_v_dim)
    sycl_core_cpu = sycl_core[:n_actual].detach().to("cpu").float()
    oracle_core_cpu = oracle_core[:n_actual].detach().to("cpu").float()
    delta = (sycl_core_cpu - oracle_core_cpu).abs()
    tol = 2e-2 + 2e-2 * oracle_core_cpu.abs()
    fail = delta > tol

    fail_count = int(fail.sum())
    total = int(fail.numel())
    max_abs = float(delta.max().item())
    mean_abs = float(delta.mean().item())

    if fail_count > 0:
        # per-v_dim fail counts: reduce over (token, head)
        per_v = fail.sum(dim=(0, 1))  # (head_v_dim,)
        v_with_fails = (per_v > 0).nonzero(as_tuple=True)[0].tolist()
        v_top = sorted(
            ((int(per_v[v]), v) for v in v_with_fails), reverse=True
        )[:6]
    else:
        v_top = []

    return dict(
        fail_count=fail_count,
        total=total,
        max_abs=max_abs,
        mean_abs=mean_abs,
        v_top=v_top,  # list of (count, v_dim)
        n_actual=n_actual,
    )


def main():
    device = torch.device("xpu")
    by_rung_layer = defaultdict(list)  # (rung, layer) -> list[stats]

    paths = sorted(CAPTURES.glob("tuple_*spec_K4_min*_max*.pt"))
    print(f"[footprint] found {len(paths)} spec captures", flush=True)

    for i, p in enumerate(paths):
        m = LAYER_RE.search(p.name)
        if not m:
            continue
        layer = int(m.group(1))
        rung = m.group(2)  # spec_K4_min1_max1 or spec_K4_min4_max4

        payload = torch.load(p, map_location="cpu", weights_only=False)
        payload["__path__"] = str(p)
        try:
            s = per_capture_stats(payload, device)
        except Exception as e:
            print(f"[{i+1}/{len(paths)}] {p.name}: ERROR {type(e).__name__}: {e}",
                  flush=True)
            continue
        if s is None:
            continue
        by_rung_layer[(rung, layer)].append(s)
        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{len(paths)}] processed", flush=True)

    print("\n=== per-layer aggregate (core_attn_out, sycl vs inline FLA oracle) ===")
    print(f"{'rung':<22} {'layer':>5} {'n':>3} "
          f"{'max':>10} {'mean':>10} {'fail':>9} {'v_top (count:v_dim)'}")
    rung_layer_v_union = defaultdict(set)
    for (rung, layer), stats_list in sorted(by_rung_layer.items()):
        n = len(stats_list)
        max_abs = max(s["max_abs"] for s in stats_list)
        mean_abs = sum(s["mean_abs"] for s in stats_list) / n
        fail_total = sum(s["fail_count"] for s in stats_list)
        cell_total = sum(s["total"] for s in stats_list)
        # top v across all captures of this (rung, layer)
        v_counts = defaultdict(int)
        for s in stats_list:
            for cnt, v in s["v_top"]:
                v_counts[v] += cnt
                rung_layer_v_union[rung].add(v)
        v_top = sorted(v_counts.items(), key=lambda kv: -kv[1])[:5]
        v_top_str = " ".join(f"{cnt}:{v}" for v, cnt in v_top)
        print(f"{rung:<22} {layer:>5} {n:>3} {max_abs:10.3e} {mean_abs:10.3e} "
              f"{fail_total:>5}/{cell_total:<5} {v_top_str}")

    print("\n=== union of failing v_dim coords (any layer, any capture) ===")
    for rung, vset in sorted(rung_layer_v_union.items()):
        print(f"{rung}: {sorted(vset)}")


if __name__ == "__main__":
    sys.exit(main() or 0)

"""Compute KL(baseline||tag) per position and summarize.

Run after decode_kl.py for baseline + each TQ mode.
Splits positions into:
  - first_chunk: [0 .. chunk_size)            — TQ-unaffected (no prior cache)
  - post_chunk:  [chunk_size .. prompt_len)   — TQ-compressed cache reads

Reports KL averaged over each band, plus top-1 / top-5 agreement rates
with the baseline.
"""

import argparse
import json
import math
from pathlib import Path
from statistics import mean


def kl_at_position(p_lps: dict, q_lps: dict) -> float:
    if not p_lps or not q_lps:
        return float("nan")
    common = set(p_lps.keys()) & set(q_lps.keys())
    if not common:
        return float("nan")
    p_vals = [p_lps[k] for k in common]
    q_vals = [q_lps[k] for k in common]
    p_max = max(p_vals); q_max = max(q_vals)
    p_unnorm = [math.exp(v - p_max) for v in p_vals]
    q_unnorm = [math.exp(v - q_max) for v in q_vals]
    p_sum = sum(p_unnorm); q_sum = sum(q_unnorm)
    if p_sum <= 0 or q_sum <= 0:
        return float("nan")
    kl = 0.0
    for pu, qu in zip(p_unnorm, q_unnorm):
        p = pu / p_sum
        q = qu / q_sum
        if p > 0:
            kl += p * (math.log(p) - math.log(q))
    return kl


def top_k_keys(lps: dict, k: int):
    return [tok for tok, _ in sorted(lps.items(), key=lambda kv: -kv[1])[:k]]


def summarize(base, other, chunk_size):
    """Compare base captures vs other captures (lists of lists of dicts)."""
    kl_first, kl_post = [], []
    top1_first, top1_post = [], []
    top5_first, top5_post = [], []
    for b_caps, o_caps in zip(base, other):
        for pos, (b, o) in enumerate(zip(b_caps, o_caps)):
            kl = kl_at_position(b, o)
            if math.isnan(kl):
                continue
            b_top1 = top_k_keys(b, 1)
            o_top1 = top_k_keys(o, 1)
            b_top5 = set(top_k_keys(b, 5))
            o_top5 = set(top_k_keys(o, 5))
            top1_match = int(bool(b_top1) and b_top1 == o_top1)
            top5_overlap = len(b_top5 & o_top5) / 5 if b_top5 else 0.0
            if pos < chunk_size:
                kl_first.append(kl)
                top1_first.append(top1_match)
                top5_first.append(top5_overlap)
            else:
                kl_post.append(kl)
                top1_post.append(top1_match)
                top5_post.append(top5_overlap)
    return {
        "kl_first_chunk_avg": mean(kl_first) if kl_first else None,
        "kl_post_chunk_avg": mean(kl_post) if kl_post else None,
        "top1_first_chunk": mean(top1_first) if top1_first else None,
        "top1_post_chunk": mean(top1_post) if top1_post else None,
        "top5_first_chunk": mean(top5_first) if top5_first else None,
        "top5_post_chunk": mean(top5_post) if top5_post else None,
        "n_first": len(kl_first),
        "n_post": len(kl_post),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--tags", nargs="+", required=True)
    args = ap.parse_args()

    base_path = args.results_dir / f"kl_dist_{args.baseline}.json"
    with open(base_path) as f:
        base_data = json.load(f)
    chunk_size = base_data["chunk_size"]
    base = base_data["captures"]

    print(f"Baseline: {args.baseline}")
    print(f"Chunk size: {chunk_size} tokens (positions < chunk = TQ-unaffected, ≥ chunk = TQ-affected)")
    print()
    print(f"{'tag':<14} {'KL pre':>10} {'KL post':>10} {'top1 pre':>10} {'top1 post':>10} {'top5 pre':>10} {'top5 post':>10}")
    print("-" * 90)

    for tag in args.tags:
        path = args.results_dir / f"kl_dist_{tag}.json"
        if not path.exists():
            print(f"  [skip] {path}: missing")
            continue
        with open(path) as f:
            cap = json.load(f)
        s = summarize(base, cap["captures"], chunk_size)
        def fmt(x, w=10, p=4):
            return f"{x:>{w}.{p}f}" if x is not None else f"{'-':>{w}}"
        print(f"{tag:<14} {fmt(s['kl_first_chunk_avg'])} {fmt(s['kl_post_chunk_avg'])} "
              f"{fmt(s['top1_first_chunk'], 10, 1) if s['top1_first_chunk'] is not None else '-':>10} "
              f"{fmt(s['top1_post_chunk'], 10, 1)}".replace(" -:.4f", "      -")
              + f" {fmt(s['top5_first_chunk'], 10, 3)} {fmt(s['top5_post_chunk'], 10, 3)}")


if __name__ == "__main__":
    main()

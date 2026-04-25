"""Compute KL divergence between two logprob captures.

Compares a baseline capture against a compressed-KV capture to measure
quality degradation from KV cache quantization.

Usage:
    python compute_kl.py results/logprobs_baseline.json results/logprobs_rotorquant-3bit.json

Outputs:
    - Per-position KL divergence (where does quality degrade most?)
    - Aggregate KL divergence
    - Perplexity comparison
    - Top-1 agreement rate (how often does the top prediction match?)
    - Cosine similarity of logprob distributions
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def load_capture(path):
    with open(path) as f:
        return json.load(f)


def kl_divergence_topk(p_logprobs, q_logprobs):
    """Approximate KL(P||Q) from top-k logprob dictionaries.

    Since we only have top-k logprobs (not the full vocab distribution),
    this computes KL over the union of top-k tokens from both distributions.
    The remaining probability mass is lumped into a "rest" bucket.
    """
    if not p_logprobs or not q_logprobs:
        return 0.0

    all_tokens = set(p_logprobs.keys()) | set(q_logprobs.keys())

    p_total = sum(math.exp(lp) for lp in p_logprobs.values())
    q_total = sum(math.exp(lp) for lp in q_logprobs.values())

    kl = 0.0
    for tok in all_tokens:
        p_lp = p_logprobs.get(tok)
        q_lp = q_logprobs.get(tok)

        if p_lp is None or q_lp is None:
            continue

        p_prob = math.exp(p_lp)
        q_prob = math.exp(q_lp)

        if p_prob > 1e-10 and q_prob > 1e-10:
            kl += p_prob * (p_lp - q_lp)

    return max(0.0, kl)


def cosine_similarity_topk(p_logprobs, q_logprobs):
    """Cosine similarity between top-k logprob vectors."""
    if not p_logprobs or not q_logprobs:
        return 1.0

    all_tokens = set(p_logprobs.keys()) | set(q_logprobs.keys())

    dot = 0.0
    p_norm = 0.0
    q_norm = 0.0

    for tok in all_tokens:
        p_val = math.exp(p_logprobs.get(tok, -100))
        q_val = math.exp(q_logprobs.get(tok, -100))
        dot += p_val * q_val
        p_norm += p_val * p_val
        q_norm += q_val * q_val

    denom = math.sqrt(p_norm) * math.sqrt(q_norm)
    return dot / denom if denom > 1e-10 else 1.0


def compare(baseline, comparison):
    b_caps = {c["sequence_id"]: c for c in baseline["captures"]}
    c_caps = {c["sequence_id"]: c for c in comparison["captures"]}

    common_ids = sorted(set(b_caps.keys()) & set(c_caps.keys()))
    if not common_ids:
        print("ERROR: No matching sequence IDs between captures")
        sys.exit(1)

    total_kl = 0.0
    total_cosine = 0.0
    total_top1_match = 0
    total_positions = 0
    kl_by_position_bucket = defaultdict(list)  # bucket by relative position

    for sid in common_ids:
        b_tokens = b_caps[sid]["token_logprobs"]
        c_tokens = c_caps[sid]["token_logprobs"]

        n = min(len(b_tokens), len(c_tokens))
        for i in range(1, n):  # skip position 0 (no conditioning context)
            b_top = b_tokens[i].get("top_logprobs", {})
            c_top = c_tokens[i].get("top_logprobs", {})

            kl = kl_divergence_topk(b_top, c_top)
            cos = cosine_similarity_topk(b_top, c_top)

            # Top-1 agreement
            b_top1 = max(b_top, key=b_top.get) if b_top else ""
            c_top1 = max(c_top, key=c_top.get) if c_top else ""
            top1_match = 1 if b_top1 == c_top1 else 0

            total_kl += kl
            total_cosine += cos
            total_top1_match += top1_match
            total_positions += 1

            # Bucket by relative position (0-25%, 25-50%, 50-75%, 75-100%)
            rel_pos = i / n
            bucket = int(rel_pos * 4)
            kl_by_position_bucket[min(bucket, 3)].append(kl)

    avg_kl = total_kl / total_positions if total_positions > 0 else 0
    avg_cosine = total_cosine / total_positions if total_positions > 0 else 0
    top1_rate = total_top1_match / total_positions if total_positions > 0 else 0

    return {
        "baseline_tag": baseline["tag"],
        "comparison_tag": comparison["tag"],
        "num_sequences": len(common_ids),
        "total_positions": total_positions,
        "avg_kl_divergence": avg_kl,
        "avg_cosine_similarity": avg_cosine,
        "top1_agreement_rate": top1_rate,
        "baseline_perplexity": baseline["perplexity"],
        "comparison_perplexity": comparison["perplexity"],
        "perplexity_delta": comparison["perplexity"] - baseline["perplexity"],
        "perplexity_ratio": comparison["perplexity"] / baseline["perplexity"] if baseline["perplexity"] > 0 else 0,
        "kl_by_position": {
            f"{b*25}-{(b+1)*25}%": sum(kls) / len(kls) if kls else 0
            for b, kls in sorted(kl_by_position_bucket.items())
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Compare logprob captures via KL divergence")
    parser.add_argument("baseline", type=Path, help="Baseline logprobs JSON")
    parser.add_argument("comparison", type=Path, help="Comparison logprobs JSON")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    baseline = load_capture(args.baseline)
    comparison = load_capture(args.comparison)

    result = compare(baseline, comparison)

    print(f"=== Quality Comparison: {result['baseline_tag']} vs {result['comparison_tag']} ===")
    print(f"Sequences: {result['num_sequences']}, Positions: {result['total_positions']}")
    print()
    print(f"  KL Divergence (avg):       {result['avg_kl_divergence']:.6f}")
    print(f"  Cosine Similarity (avg):   {result['avg_cosine_similarity']:.6f}")
    print(f"  Top-1 Agreement Rate:      {result['top1_agreement_rate']:.4f} ({result['top1_agreement_rate']*100:.1f}%)")
    print()
    print(f"  Baseline Perplexity:       {result['baseline_perplexity']:.4f}")
    print(f"  Comparison Perplexity:     {result['comparison_perplexity']:.4f}")
    print(f"  Perplexity Delta:          {result['perplexity_delta']:+.4f}")
    print(f"  Perplexity Ratio:          {result['perplexity_ratio']:.4f}x")
    print()
    print("  KL by sequence position:")
    for bucket, kl in result["kl_by_position"].items():
        bar = "#" * int(kl * 1000)
        print(f"    {bucket:>8}: {kl:.6f} {bar}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

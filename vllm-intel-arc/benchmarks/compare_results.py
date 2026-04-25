"""Compare benchmark results across configurations.

Usage:
    python benchmarks/compare_results.py results/baseline.json results/with-rotorquant.json
"""

import argparse
import json
import sys
from pathlib import Path


def load_results(path):
    with open(path) as f:
        return json.load(f)


def compare(baseline_path, comparison_path):
    baseline = load_results(baseline_path)
    comparison = load_results(comparison_path)

    baseline_by_key = {}
    for r in baseline:
        key = (r["prompt_len"], r["output_len"], r["concurrency"])
        baseline_by_key[key] = r

    print(f"{'Config':<20} {'P':>5} {'O':>5} {'C':>3} | "
          f"{'tok/s':>8} {'TTFT':>8} {'p50':>8} | "
          f"{'speedup':>8} {'TTFT_delta':>10}")
    print("-" * 100)

    for r in comparison:
        key = (r["prompt_len"], r["output_len"], r["concurrency"])
        b = baseline_by_key.get(key)
        if not b:
            continue

        speedup = r["throughput_tps"] / b["throughput_tps"] if b["throughput_tps"] > 0 else 0
        ttft_delta = r["ttft_ms"] - b["ttft_ms"]

        print(f"{r['config']:<20} {r['prompt_len']:>5} {r['output_len']:>5} {r['concurrency']:>3} | "
              f"{r['throughput_tps']:>8.1f} {r['ttft_ms']:>7.0f}ms {r['p50_latency_ms']:>7.0f}ms | "
              f"{speedup:>7.2f}x {ttft_delta:>+9.0f}ms")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("comparison", type=Path)
    args = parser.parse_args()

    compare(args.baseline, args.comparison)


if __name__ == "__main__":
    main()

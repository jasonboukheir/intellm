"""Compare greedy decode outputs across KV-cache modes.

Reads decode_<tag>.json captures from decode_quality.py and reports per-tag:
  - Exact prefix match length (how many tokens before diverging from baseline)
  - Token-level Hamming distance (positions where tag differs from baseline)
  - Per-token NLL drift (sum of |NLL_tag - NLL_baseline|)
  - Per-token NLL of tag's own choices

Lower-quality modes show shorter exact-prefix and higher Hamming.
"""

import argparse
import json
from pathlib import Path
from statistics import mean


def load(path: Path):
    with open(path) as f:
        return json.load(f)


def prefix_match(a: list, b: list) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            return n
        n += 1
    return n


def hamming(a: list, b: list) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--tags", nargs="+", required=True)
    args = ap.parse_args()

    base = load(args.results_dir / f"decode_{args.baseline}.json")
    base_results = base["results"]

    print(f"Baseline: {args.baseline}  prompts={len(base_results)}  decoded={base['total_decoded_tokens']}  NLL={base['avg_nll']:.4f}  PPL={base['perplexity']:.4f}")
    print()
    print(f"{'tag':<14} {'PPL':>8} {'NLL':>8} {'prefix-match':>13} {'hamming-frac':>14} {'first-3-prompt-divergence':>26}")
    print("-" * 90)
    print(f"{args.baseline:<14} {base['perplexity']:>8.4f} {base['avg_nll']:>8.4f} {'(self)':>13} {'(self)':>14} {'(self)':>26}")

    for tag in args.tags:
        path = args.results_dir / f"decode_{tag}.json"
        if not path.exists():
            print(f"  [skip] {path}: missing")
            continue
        cap = load(path)
        results = cap["results"]
        prefix_lens = []
        hammings = []
        per_prompt_div = []
        for b, c in zip(base_results, results):
            b_toks = b["tokens"]
            c_toks = c["tokens"]
            n = min(len(b_toks), len(c_toks))
            pm = prefix_match(b_toks[:n], c_toks[:n])
            hm = hamming(b_toks[:n], c_toks[:n])
            prefix_lens.append(pm / n if n else 1.0)
            hammings.append(hm / n if n else 0.0)
            per_prompt_div.append(pm)
        first3 = ",".join(str(d) for d in per_prompt_div[:3])
        print(f"{tag:<14} {cap['perplexity']:>8.4f} {cap['avg_nll']:>8.4f} {mean(prefix_lens):>12.1%}  {mean(hammings):>13.1%}  {first3:>26}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Approximate KL divergence between two top-K logprob captures.

Inputs are two JSON files produced by capture_logprobs.py. We treat each
position as a discrete distribution over a token vocabulary, but we only
have the top-K entries — so this is *truncated* KL with explicit handling
of the unmodeled tail.

For each prompt × position, we:
  1. Take the union of token ids that appear in either capture's top-K.
  2. For ids missing from one side, use a tail-probability estimate
     ε = (1 - sum(top_k_probs)) / (V - K) where V is the vocab size.
  3. Compute KL(P || Q) = Σ_i P_i * log(P_i / Q_i)

Where P is the *baseline* (first arg) and Q is the *quantized* (second).

Important caveats:
  - Truncated-tail KL is a proxy. The papers (TurboQuant/IsoQuant/RotorQuant)
    typically report PPL on a held-out set, which directly uses the model's
    log-prob of the *true* next token. We do that too as a secondary metric.
  - Two captures only align if they generated the same tokens (we use
    greedy temperature=0 so they often do, but quants may diverge). We
    align on prefix overlap and stop at first divergence per prompt; the
    number of compared positions varies per prompt and is reported.

Usage:
  python compute_kl.py results/logprobs/fp16.json results/logprobs/q8_0.json
"""

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean, median


def load(p: Path) -> dict:
    """Load a capture and re-key top_logprobs dicts back to int.

    JSON serialization stringifies int keys, which would cause every
    `picked_id in top_logprobs` lookup to silently miss and fall through
    to the tail-mass estimate — corrupting both KL and CE numbers.
    """
    data = json.loads(p.read_text())
    for prompt in data.get("prompts", []):
        for tok in prompt.get("tokens", []):
            tlp = tok.get("top_logprobs") or {}
            tok["top_logprobs"] = {int(k): v for k, v in tlp.items()}
    return data


def softmax_check(top: dict) -> float:
    """Sum of probs in a top-K dict (used to estimate tail mass)."""
    if not top:
        return 0.0
    return sum(math.exp(v["logprob"]) for v in top.values())


def kl_at_position(top_p: dict, top_q: dict, vocab_size: int = 248320) -> float:
    """Truncated KL(P || Q) between two top-K dicts.

    For ids in P's top-K but missing in Q's, we substitute Q's tail estimate
    ε_q = (1 - sum(Q.top)) / (V - |Q.top|) (and vice versa for P).
    Tail mass at non-top-K ids is dropped from the sum (their contribution
    to KL is small when both sides agree it's tail; when they disagree
    sharply we'd miss it — true full-vocab KL would catch that).
    """
    p_sum = softmax_check(top_p)
    q_sum = softmax_check(top_q)
    p_tail_per_token = max(1e-30, (1.0 - p_sum) / max(1, vocab_size - len(top_p)))
    q_tail_per_token = max(1e-30, (1.0 - q_sum) / max(1, vocab_size - len(top_q)))

    ids = set(top_p) | set(top_q)
    kl = 0.0
    for i in ids:
        p = math.exp(top_p[i]["logprob"]) if i in top_p else p_tail_per_token
        q = math.exp(top_q[i]["logprob"]) if i in top_q else q_tail_per_token
        if p <= 0 or q <= 0:
            continue
        kl += p * math.log(p / q)
    return max(0.0, kl)


def cross_entropy_at_position(picked_id: int, top: dict, vocab_size: int = 248320) -> float:
    """-log Q(picked_id) from a top-K capture; use tail estimate if missing."""
    if picked_id in top:
        return -top[picked_id]["logprob"]
    q_sum = softmax_check(top)
    q_tail = max(1e-30, (1.0 - q_sum) / max(1, vocab_size - len(top)))
    return -math.log(q_tail)


def compare(p_data: dict, q_data: dict, vocab_size: int = 248320) -> dict:
    if len(p_data["prompts"]) != len(q_data["prompts"]):
        sys.exit(
            f"prompt-count mismatch: {len(p_data['prompts'])} vs {len(q_data['prompts'])}"
        )

    rows = []
    for i, (pp, qp) in enumerate(zip(p_data["prompts"], q_data["prompts"])):
        if pp["prompt"] != qp["prompt"]:
            sys.exit(f"prompt #{i} text differs between captures")

        # Align on prefix of generated token ids; stop at first divergence.
        n = min(len(pp["tokens"]), len(qp["tokens"]))
        agree = 0
        for k in range(n):
            if pp["tokens"][k]["id"] == qp["tokens"][k]["id"]:
                agree += 1
            else:
                break

        # Only compare positions where greedy *agreed* — past divergence the two
        # captures condition on different prefixes, so the distributions describe
        # different things. (Real KL/PPL eval would teacher-force the same prefix;
        # that's a phase-2b upgrade.)
        kls = []
        ce_q_picks_p = []
        for k in range(agree):
            kls.append(kl_at_position(
                pp["tokens"][k]["top_logprobs"], qp["tokens"][k]["top_logprobs"], vocab_size,
            ))
            ce_q_picks_p.append(cross_entropy_at_position(
                pp["tokens"][k]["id"], qp["tokens"][k]["top_logprobs"], vocab_size,
            ))

        rows.append({
            "prompt_idx": i,
            "n_compared": n,
            "n_token_agreement": agree,
            "kl_mean": mean(kls) if kls else 0.0,
            "kl_max": max(kls) if kls else 0.0,
            "kl_p50": median(kls) if kls else 0.0,
            "ce_qpicksp_mean": mean(ce_q_picks_p) if ce_q_picks_p else 0.0,
        })

    n_prompts = len(rows)
    total_compared = sum(r["n_compared"] for r in rows)
    total_agreed   = sum(r["n_token_agreement"] for r in rows)
    summary = {
        "p_tag": p_data.get("tag"),
        "q_tag": q_data.get("tag"),
        "p_kv": (p_data.get("kv_cache_type_k"), p_data.get("kv_cache_type_v")),
        "q_kv": (q_data.get("kv_cache_type_k"), q_data.get("kv_cache_type_v")),
        "n_prompts": n_prompts,
        "agreement_total":    total_agreed / max(1, total_compared),
        "agreement_min_per_prompt": min(r["n_token_agreement"] / max(1, r["n_compared"]) for r in rows),
        "n_prompts_full_agreement": sum(1 for r in rows if r["n_token_agreement"] == r["n_compared"]),
        "kl_mean_of_means":   mean(r["kl_mean"] for r in rows),
        "kl_max_overall":     max(r["kl_max"]  for r in rows),
        "kl_p50_of_p50s":     median(r["kl_p50"] for r in rows),
        "ce_qpicksp_mean":    mean(r["ce_qpicksp_mean"] for r in rows),
        "ce_baseline_mean":   None,  # filled below
    }

    # Baseline cross-entropy under the BASELINE distribution itself (i.e.,
    # NLL of greedy choices under the model). Useful sanity check —
    # ce_qpicksp - ce_baseline ≈ KL of greedy-token probability.
    base_ce = []
    for pp in p_data["prompts"]:
        for tok in pp["tokens"]:
            base_ce.append(-tok["logprob"])
    summary["ce_baseline_mean"] = mean(base_ce) if base_ce else 0.0

    return {"summary": summary, "per_prompt": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", type=Path, help="JSON from capture_logprobs.py — treated as P")
    ap.add_argument("quantized", type=Path, help="JSON from capture_logprobs.py — treated as Q")
    ap.add_argument("--vocab-size", type=int, default=248320, help="Qwen3.6 vocab default")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    p = load(args.baseline)
    q = load(args.quantized)
    result = compare(p, q, vocab_size=args.vocab_size)
    s = result["summary"]

    print(f"=== KL: {s['p_tag']} (P) vs {s['q_tag']} (Q) ===")
    print(f"  Prompts:                  {s['n_prompts']}")
    print(f"  Full-agreement prompts:   {s['n_prompts_full_agreement']}/{s['n_prompts']}")
    print(f"  Total greedy agreement:   {s['agreement_total']:.3f}  (worst-prompt: {s['agreement_min_per_prompt']:.3f})")
    print(f"  KL(P||Q) per agreed-tok:  mean={s['kl_mean_of_means']:.5f}  p50={s['kl_p50_of_p50s']:.5f}  max={s['kl_max_overall']:.5f}")
    print(f"  -log Q(picked under P):   {s['ce_qpicksp_mean']:.4f}  (baseline -log P(picked)={s['ce_baseline_mean']:.4f})")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

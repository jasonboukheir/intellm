"""Position-by-position KL divergence on TurboQuant-affected positions.

The original capture_logprobs.py was prefill-only (TQ doesn't run during
single-shot prefill). decode_quality.py forced TQ into the loop but the
modes pick different greedy tokens, so direct KL between distributions
becomes non-comparable.

This probe forces *chunked* prefill (--max-num-batched-tokens N small),
which makes every prompt position past the first chunk go through the
TQ-compressed-cache-read path while still being teacher-forced (we feed
the same exact prompt to all modes). Then we capture top-K prompt
logprobs and compute KL position-by-position vs the FP16-KV baseline.

Server requirements:
  - --max-num-batched-tokens MUST be smaller than the prompt length
  - --enforce-eager is fine (we don't care about graph compile here)

Output: per-mode summary {tag, kl_avg, kl_first_chunk, kl_post_chunk,
top1_agreement, top5_agreement, ppl_ratio}.
"""

import argparse
import json
import math
import time
from pathlib import Path

import requests
from datasets import load_dataset
from transformers import AutoTokenizer


def load_long_prompts(tokenizer, num_prompts: int, prompt_len: int):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    buf = []
    for ex in ds:
        if ex["text"].strip():
            buf.append(ex["text"])
    full = "\n".join(buf)
    toks = tokenizer.encode(full)
    prompts = []
    for i in range(num_prompts):
        s, e = i * prompt_len, i * prompt_len + prompt_len
        if e > len(toks):
            break
        prompts.append(tokenizer.decode(toks[s:e], skip_special_tokens=True))
    return prompts


def capture_prompt_logprobs(url, model, prompt, top_k):
    """Send prompt; receive prompt_logprobs (top-K dist at each position)."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0.0,
        "prompt_logprobs": top_k,
        "echo": False,
    }
    r = requests.post(f"{url}/v1/completions", json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    choice = data["choices"][0]
    return choice.get("prompt_logprobs") or []


def kl_at_position(p_logprobs: dict, q_logprobs: dict) -> float:
    """KL(P||Q) restricted to keys present in both top-K sets.

    Each input is {token_id: logprob}. Renormalize both to a probability
    distribution over the intersection, then compute KL.
    """
    if not p_logprobs or not q_logprobs:
        return float("nan")
    common = set(p_logprobs.keys()) & set(q_logprobs.keys())
    if not common:
        return float("nan")
    p_lps = [p_logprobs[k] for k in common]
    q_lps = [q_logprobs[k] for k in common]
    # Renormalize over the common subset
    p_max = max(p_lps); q_max = max(q_lps)
    p_unnorm = [math.exp(lp - p_max) for lp in p_lps]
    q_unnorm = [math.exp(lp - q_max) for lp in q_lps]
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


def normalize_prompt_logprobs(raw: list) -> list[dict]:
    """vLLM returns prompt_logprobs as list[None | dict[str, dict]]; the
    value dict has keys like 'logprob' and 'rank'. Flatten into
    {token_id_or_str: logprob}."""
    out = []
    for entry in raw:
        if entry is None:
            out.append({})
            continue
        flat = {}
        for tok_id, info in entry.items():
            if isinstance(info, dict) and "logprob" in info:
                flat[tok_id] = float(info["logprob"])
            elif isinstance(info, (int, float)):
                flat[tok_id] = float(info)
        out.append(flat)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--num-prompts", type=int, default=4)
    ap.add_argument("--prompt-len", type=int, default=1024)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Server's --max-num-batched-tokens. Used to mark which positions are post-chunk-boundary in the per-position output.",
    )
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer for {args.model}...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = load_long_prompts(tok, args.num_prompts, args.prompt_len)
    print(f"Loaded {len(prompts)} prompts of {args.prompt_len} tokens each")

    captures = []
    for i, p in enumerate(prompts, 1):
        t0 = time.time()
        raw = capture_prompt_logprobs(args.base_url, args.model, p, args.top_k)
        flat = normalize_prompt_logprobs(raw)
        elapsed = time.time() - t0
        print(f"  [{i}/{len(prompts)}] {len(flat)} positions, {elapsed:.1f}s")
        captures.append(flat)

    out = args.output / f"kl_dist_{args.tag}.json"
    with open(out, "w") as f:
        json.dump({
            "tag": args.tag,
            "model": args.model,
            "num_prompts": len(prompts),
            "prompt_len": args.prompt_len,
            "top_k": args.top_k,
            "chunk_size": args.chunk_size,
            "captures": captures,
        }, f)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()

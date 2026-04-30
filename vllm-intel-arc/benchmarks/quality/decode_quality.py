"""Greedy-generation quality probe for KV-cache compression methods.

The standard `capture_logprobs.py` uses prefill-only logprobs (echo=true,
max_tokens=1), which doesn't exercise the TurboQuant *decode* path — TQ
attention runs uncompressed K/V during prefill and only reads compressed
KV during decode. So all TQ modes look identical under that probe.

This script forces TQ into the loop:
  1. Prefill a long prompt (≥ prefill_len tokens).
  2. Greedy-decode `gen_len` more tokens with logprobs enabled.
  3. Each decode step reads K/V from the compressed cache built up by
     earlier steps — so TQ degradation accumulates across the generation.
  4. Save the generated text + per-step top-1 logprob.

Run once per mode (baseline, k8v4, 4bit_nc, …) and compare:
  - Greedy text divergence: how soon does the output deviate?
  - Token-level NLL: average -log P(generated token) — diverges if TQ
    miscalibrates the distribution.
"""

import argparse
import json
import math
import time
from pathlib import Path

import requests
from datasets import load_dataset
from transformers import AutoTokenizer


def load_wikitext_chunks(tokenizer, num_prompts: int, prefill_len: int):
    """Pull `num_prompts` prefill_len-token chunks from wikitext-2 test.

    Concatenate the dataset's article texts into a stream and slice into
    fixed-length token chunks — this avoids the "most articles are short"
    filter problem.
    """
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    buf = []
    for ex in ds:
        if ex["text"].strip():
            buf.append(ex["text"])
    full = "\n".join(buf)
    toks = tokenizer.encode(full)
    prompts = []
    for i in range(num_prompts):
        start = i * prefill_len
        end = start + prefill_len
        if end > len(toks):
            break
        prompts.append(tokenizer.decode(toks[start:end], skip_special_tokens=True))
    return prompts


def greedy_decode(url, model, prompt, gen_len: int):
    """Greedy-decode `gen_len` tokens; return text + per-step top-1 logprob."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": gen_len,
        "temperature": 0.0,
        "logprobs": 1,
        "echo": False,
    }
    resp = requests.post(f"{url}/v1/completions", json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    choice = data["choices"][0]
    text = choice["text"]
    lps = choice.get("logprobs", {})
    token_logprobs = lps.get("token_logprobs") or []
    tokens = lps.get("tokens") or []
    return {
        "text": text,
        "tokens": tokens,
        "token_logprobs": [lp for lp in token_logprobs if lp is not None],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--output", type=Path, default=Path("results/quality"))
    ap.add_argument("--num-prompts", type=int, default=8)
    ap.add_argument("--prefill-len", type=int, default=512)
    ap.add_argument("--gen-len", type=int, default=128)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer for {args.model}...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"Loading {args.num_prompts} chunks of {args.prefill_len} tokens from wikitext-2...")
    prompts = load_wikitext_chunks(tok, args.num_prompts, args.prefill_len)
    print(f"Loaded {len(prompts)} prompts")

    results = []
    nll_sum = 0.0
    nll_count = 0
    t0 = time.time()
    for i, p in enumerate(prompts, 1):
        r = greedy_decode(args.base_url, args.model, p, args.gen_len)
        nll = -sum(r["token_logprobs"])
        n = len(r["token_logprobs"])
        nll_sum += nll
        nll_count += n
        per_tok_nll = nll / max(1, n)
        ppl = math.exp(per_tok_nll)
        print(f"  [{i}/{len(prompts)}] {n} tokens, ppl={ppl:.3f}")
        results.append({
            "prompt_preview": p[-200:],
            "generated": r["text"],
            "tokens": r["tokens"],
            "token_logprobs": r["token_logprobs"],
            "nll": nll,
            "n": n,
        })

    overall_per_tok_nll = nll_sum / max(1, nll_count)
    overall_ppl = math.exp(overall_per_tok_nll)
    elapsed = time.time() - t0
    summary = {
        "tag": args.tag,
        "model": args.model,
        "num_prompts": len(prompts),
        "prefill_len": args.prefill_len,
        "gen_len": args.gen_len,
        "total_decoded_tokens": nll_count,
        "avg_nll": overall_per_tok_nll,
        "perplexity": overall_ppl,
        "elapsed_s": elapsed,
        "results": results,
    }
    out = args.output / f"decode_{args.tag}.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Decode tokens: {nll_count}")
    print(f"Avg NLL: {overall_per_tok_nll:.4f}")
    print(f"Perplexity: {overall_ppl:.4f}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()

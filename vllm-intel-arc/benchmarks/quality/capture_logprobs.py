"""Capture per-token logprobs from a vLLM server for quality evaluation.

Feeds a fixed eval dataset through the model and saves the full logprob
response for each token position. This is the "measurement" step — run it
once for baseline (BF16 KV cache) and again for each compression method.

Usage:
    # Baseline capture
    python capture_logprobs.py --tag baseline --output results/

    # After enabling RotorQuant
    python capture_logprobs.py --tag rotorquant-3bit --output results/

Then compare with compute_kl.py.
"""

import argparse
import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import requests
from datasets import load_dataset
from transformers import AutoTokenizer


@dataclass
class TokenLogprob:
    position: int
    token: str
    token_id: int
    logprob: float
    top_logprobs: dict = field(default_factory=dict)  # {token: logprob}


@dataclass
class SequenceCapture:
    sequence_id: int
    text: str
    num_tokens: int
    token_logprobs: list = field(default_factory=list)
    total_nll: float = 0.0


def load_wikitext2(tokenizer, max_sequences=100, max_tokens_per_seq=512):
    """Load WikiText-2 test set, split into sequences."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    sequences = []
    current_text = ""

    for item in dataset:
        text = item["text"].strip()
        if not text:
            continue

        current_text += " " + text if current_text else text
        tokens = tokenizer.encode(current_text)

        if len(tokens) >= max_tokens_per_seq:
            # Truncate to max_tokens and decode back for clean sequence
            truncated = tokenizer.decode(tokens[:max_tokens_per_seq], skip_special_tokens=True)
            sequences.append(truncated)
            current_text = ""

            if len(sequences) >= max_sequences:
                break

    return sequences


def load_pg19_sample(tokenizer, max_sequences=50, max_tokens_per_seq=1024):
    """Load PG-19 test set samples for long-context evaluation."""
    dataset = load_dataset("pg19", split="test", streaming=True)

    sequences = []
    for item in dataset:
        text = item["text"][:8000]  # rough char limit
        tokens = tokenizer.encode(text)
        if len(tokens) >= max_tokens_per_seq:
            truncated = tokenizer.decode(tokens[:max_tokens_per_seq], skip_special_tokens=True)
            sequences.append(truncated)
            if len(sequences) >= max_sequences:
                break

    return sequences


def capture_sequence(url, model, text, logprobs_k=20):
    """Send text to vLLM and capture per-token logprobs.

    Uses the completions endpoint with echo=true and max_tokens=1 to get
    logprobs for all prompt tokens plus one generated token.

    vLLM returns prompt_logprobs when echo is enabled.
    """
    payload = {
        "model": model,
        "prompt": text,
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": logprobs_k,
        "echo": True,
    }

    resp = requests.post(f"{url}/v1/completions", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    choice = data["choices"][0]
    logprobs_data = choice.get("logprobs", {})

    token_logprobs = []
    tokens = logprobs_data.get("tokens", [])
    token_lps = logprobs_data.get("token_logprobs", [])
    top_lps = logprobs_data.get("top_logprobs", [])

    total_nll = 0.0
    valid_count = 0

    for i, (tok, lp) in enumerate(zip(tokens, token_lps)):
        top_k = top_lps[i] if i < len(top_lps) and top_lps[i] else {}
        tl = TokenLogprob(
            position=i,
            token=tok,
            token_id=-1,  # not exposed in completions API
            logprob=lp if lp is not None else 0.0,
            top_logprobs=top_k,
        )
        token_logprobs.append(tl)
        if lp is not None:
            total_nll -= lp
            valid_count += 1

    return SequenceCapture(
        sequence_id=0,
        text=text[:200] + "..." if len(text) > 200 else text,
        num_tokens=len(tokens),
        token_logprobs=token_logprobs,
        total_nll=total_nll,
    )


def main():
    parser = argparse.ArgumentParser(description="Capture logprobs for quality evaluation")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default=None, help="Model name (auto-detected if not set)")
    parser.add_argument("--tag", required=True, help="Label for this capture (e.g. 'baseline', 'rotorquant-3bit')")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--dataset", choices=["wikitext2", "pg19"], default="wikitext2")
    parser.add_argument("--max-sequences", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--logprobs-k", type=int, default=20, help="Top-k logprobs to capture")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Auto-detect model
    if args.model is None:
        models = requests.get(f"{args.base_url}/v1/models").json()
        args.model = models["data"][0]["id"]
        print(f"Auto-detected model: {args.model}")

    # Load tokenizer for splitting eval data
    print(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Load eval dataset
    print(f"Loading {args.dataset} (max {args.max_sequences} sequences, {args.max_tokens} tokens each)...")
    if args.dataset == "wikitext2":
        sequences = load_wikitext2(tokenizer, args.max_sequences, args.max_tokens)
    else:
        sequences = load_pg19_sample(tokenizer, args.max_sequences, args.max_tokens)
    print(f"Loaded {len(sequences)} sequences")

    # Capture logprobs
    captures = []
    total_nll = 0.0
    total_tokens = 0

    for i, text in enumerate(sequences):
        print(f"  [{i+1}/{len(sequences)}] ", end="", flush=True)
        start = time.perf_counter()
        cap = capture_sequence(args.base_url, args.model, text, args.logprobs_k)
        cap.sequence_id = i
        elapsed = time.perf_counter() - start
        captures.append(cap)

        total_nll += cap.total_nll
        total_tokens += cap.num_tokens
        ppl_so_far = 2 ** (total_nll / total_tokens / 0.6931471805599453) if total_tokens > 0 else float("inf")
        print(f"{cap.num_tokens} tokens, {elapsed:.1f}s, running ppl={ppl_so_far:.2f}")

    # Compute final perplexity: exp(avg NLL)
    import math
    avg_nll = total_nll / total_tokens if total_tokens > 0 else 0
    perplexity = math.exp(avg_nll)

    result = {
        "tag": args.tag,
        "model": args.model,
        "dataset": args.dataset,
        "num_sequences": len(captures),
        "total_tokens": total_tokens,
        "perplexity": perplexity,
        "avg_nll": avg_nll,
        "captures": [asdict(c) for c in captures],
    }

    outfile = args.output / f"logprobs_{args.tag}.json"
    with open(outfile, "w") as f:
        json.dump(result, f)
    print(f"\nPerplexity: {perplexity:.4f}")
    print(f"Avg NLL: {avg_nll:.4f}")
    print(f"Saved to {outfile} ({outfile.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

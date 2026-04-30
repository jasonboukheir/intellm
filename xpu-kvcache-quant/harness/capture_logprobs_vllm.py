#!/usr/bin/env python3
"""vLLM-shaped capture matching llamacpp-intel-arc/harness/capture_logprobs.py output.

vLLM's /v1/completions returns OpenAI-standard logprobs (string-keyed top_logprobs,
no token IDs). We enable two vLLM extensions to recover IDs:
  - return_tokens_as_token_ids=True  → tokens become "token_id:{N}" strings
  - return_token_ids=True            → choice.token_ids is the picked-token id list

Output schema is identical to capture_logprobs.py so compute_kl.py runs unchanged.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests


TOKEN_ID_PREFIX = "token_id:"


def parse_id(token_str: str) -> int:
    if not token_str.startswith(TOKEN_ID_PREFIX):
        raise ValueError(f"expected '{TOKEN_ID_PREFIX}N' marker, got {token_str!r}")
    return int(token_str[len(TOKEN_ID_PREFIX):])


def fetch_model(url: str) -> str:
    r = requests.get(f"{url}/v1/models", timeout=10)
    r.raise_for_status()
    return r.json()["data"][0]["id"]


def capture_one(url: str, model: str, prompt: str, max_tokens: int, top_k: int) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "logprobs": top_k,
        "return_tokens_as_token_ids": True,
        "return_token_ids": True,
    }
    r = requests.post(f"{url}/v1/completions", json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    choice = data["choices"][0]
    lp = choice.get("logprobs") or {}
    tokens = lp.get("tokens") or []                     # ["token_id:N", ...]
    token_logprobs = lp.get("token_logprobs") or []     # [-0.12, ...]
    top_logprobs = lp.get("top_logprobs") or []         # [{"token_id:N": -0.12, ...}, ...]
    picked_token_ids = choice.get("token_ids") or []    # [N, N, ...]

    out = []
    for i, marker in enumerate(tokens):
        picked_id = picked_token_ids[i] if i < len(picked_token_ids) else parse_id(marker)
        top = {}
        if i < len(top_logprobs) and top_logprobs[i]:
            for k, v in top_logprobs[i].items():
                tok_id = parse_id(k)
                top[tok_id] = {"token": k, "logprob": v}
        out.append({
            "id": picked_id,
            "token": marker,
            "logprob": token_logprobs[i] if i < len(token_logprobs) else None,
            "top_logprobs": top,
        })
    return {
        "prompt": prompt,
        "completion": choice.get("text", ""),
        "finish_reason": choice.get("finish_reason"),
        "tokens": out,
        "timings": {},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--kv-cache-dtype", default=None,
                    help="Recorded into the JSON for downstream labelling; not sent to the server.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    prompts = [
        line.strip()
        for line in args.prompts.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not prompts:
        sys.exit("no prompts loaded")

    model = fetch_model(args.base_url)

    out = {
        "tag": args.tag,
        "model": model,
        "kv_cache_type_k": args.kv_cache_dtype,
        "kv_cache_type_v": args.kv_cache_dtype,
        "max_tokens": args.max_tokens,
        "top_k": args.top_k,
        "prompts": [],
    }

    print(f"=== capture_logprobs_vllm ===")
    print(f"  url:           {args.base_url}")
    print(f"  model:         {model}")
    print(f"  kv_cache_dtype: {args.kv_cache_dtype}")
    print(f"  prompts:       {len(prompts)}")
    print(f"  out:           {args.out}")

    t0 = time.perf_counter()
    for i, p in enumerate(prompts, 1):
        print(f"  [{i:2d}/{len(prompts)}] {p[:60]!r}...", end=" ", flush=True)
        r = capture_one(args.base_url, model, p, args.max_tokens, args.top_k)
        print(f"{len(r['tokens'])} tokens")
        out["prompts"].append(r)

    out["wall_seconds"] = time.perf_counter() - t0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out} ({args.out.stat().st_size / 1024:.1f} KB) in {out['wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Capture top-K logprobs at each generated position for a fixed prompt set.

Usage:
  python capture_logprobs.py \
      --base-url http://127.0.0.1:8081 \
      --prompts configs/prompts/kl_test_set.txt \
      --max-tokens 64 \
      --top-k 50 \
      --tag fp16 \
      --out results/logprobs/fp16.json

The output JSON has the form:
  {
    "tag": "fp16",
    "kv_cache_type_k": "f16",
    "kv_cache_type_v": "f16",
    "model": "qwen3.6",
    "prompts": [
      {
        "prompt": "...",
        "tokens": [
          {
            "token": "...",
            "logprob": -0.12,
            "top_logprobs": {"the": -0.12, "a": -2.3, ...}
          },
          ...
        ]
      },
      ...
    ]
  }

Notes:
  - greedy decoding (temperature=0) so the *generated* token sequence is
    deterministic for a given KV-quant. Across quants the sequence may
    diverge, which is itself signal — we re-align on prefix overlap when
    computing KL.
  - top-K (default 50) means we approximate KL by treating untracked
    tail-mass as one bucket. For Qwen3.6's 248K vocab this is
    aggressive but standard practice in the rotation-quant literature.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests


def fetch_props(url: str) -> dict:
    """Pull /props from llama-server: contains cache_type_k / cache_type_v."""
    try:
        r = requests.get(f"{url}/props", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"warning: /props failed: {e}", file=sys.stderr)
        return {}


def fetch_model(url: str) -> str:
    r = requests.get(f"{url}/v1/models", timeout=5)
    r.raise_for_status()
    return r.json()["data"][0]["id"]


def capture_one(url: str, model: str, prompt: str, max_tokens: int, top_k: int) -> dict:
    """One /v1/completions call with logprobs enabled.

    llama-server returns chat-completions-style logprobs:
      logprobs.content[i] = {
        id, token, bytes, logprob,
        top_logprobs: [ {id, token, bytes, logprob}, ... ]   # length up to top_k
      }
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "logprobs": top_k,
    }
    r = requests.post(f"{url}/v1/completions", json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    choice = data["choices"][0]
    lp = choice.get("logprobs") or {}
    content = lp.get("content") or []

    out = []
    for entry in content:
        # Token-id-keyed dict makes alignment across runs simple — same id ⇒ same token,
        # avoids string-encoding ambiguity (e.g. " Paris" with leading space).
        top = {
            tl["id"]: {"token": tl.get("token"), "logprob": tl["logprob"]}
            for tl in entry.get("top_logprobs", [])
        }
        out.append({
            "id": entry.get("id"),
            "token": entry.get("token"),
            "logprob": entry.get("logprob"),
            "top_logprobs": top,
        })
    return {
        "prompt": prompt,
        "completion": choice.get("text", ""),
        "finish_reason": choice.get("finish_reason"),
        "tokens": out,
        "timings": data.get("timings", {}),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8081")
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--tag", required=True, help="label written into output (e.g. 'fp16', 'q8_0')")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    prompts = [
        line.strip()
        for line in args.prompts.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not prompts:
        sys.exit("no prompts loaded")

    props = fetch_props(args.base_url)
    model = fetch_model(args.base_url)

    out = {
        "tag": args.tag,
        "model": model,
        "kv_cache_type_k": props.get("default_generation_settings", {}).get("cache_type_k") or props.get("cache_type_k"),
        "kv_cache_type_v": props.get("default_generation_settings", {}).get("cache_type_v") or props.get("cache_type_v"),
        "max_tokens": args.max_tokens,
        "top_k": args.top_k,
        "prompts": [],
    }

    print(f"=== capture_logprobs ===")
    print(f"  url:    {args.base_url}")
    print(f"  model:  {model}")
    print(f"  k/v:    {out['kv_cache_type_k']}/{out['kv_cache_type_v']}")
    print(f"  prompts: {len(prompts)}")
    print(f"  out:    {args.out}")

    t0 = time.perf_counter()
    for i, p in enumerate(prompts, 1):
        print(f"  [{i:2d}/{len(prompts)}] {p[:60]!r}...", end=" ", flush=True)
        r = capture_one(args.base_url, model, p, args.max_tokens, args.top_k)
        n = len(r["tokens"])
        gen_tps = r["timings"].get("predicted_per_second", 0)
        print(f"{n} tokens, {gen_tps:.1f} tok/s")
        out["prompts"].append(r)

    out["wall_seconds"] = time.perf_counter() - t0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out} ({args.out.stat().st_size / 1024:.1f} KB) in {out['wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()

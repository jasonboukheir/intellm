#!/usr/bin/env python3
"""Slice configs/prompts/long_corpus.txt into prefixes of target token lengths.

We use the running llama-server's /tokenize endpoint so prefix lengths are
exact (not word-count approximations) and use the model's actual tokenizer.

Output: configs/prompts/long_corpus_<N>tok.txt (one prompt per file, single
line for compatibility with capture_logprobs.py's prompt loader).

Usage:
  python build_long_prompts.py --base-url http://127.0.0.1:8081 \
      --corpus configs/prompts/long_corpus.txt \
      --lengths 256 1024 4096 \
      --out-dir configs/prompts
"""

import argparse
import sys
from pathlib import Path

import requests


def tokenize(url: str, text: str) -> list[int]:
    r = requests.post(f"{url}/tokenize", json={"content": text}, timeout=60)
    r.raise_for_status()
    return r.json()["tokens"]


def detokenize(url: str, ids: list[int]) -> str:
    r = requests.post(f"{url}/detokenize", json={"tokens": ids}, timeout=60)
    r.raise_for_status()
    return r.json().get("content", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8081")
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--lengths", type=int, nargs="+", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    text = args.corpus.read_text()
    full_ids = tokenize(args.base_url, text)
    print(f"Corpus: {len(text):,} chars, {len(full_ids):,} tokens")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for L in args.lengths:
        if L > len(full_ids):
            print(f"  skip {L}: corpus only {len(full_ids)} tokens"); continue
        prefix = full_ids[:L]
        text_pref = detokenize(args.base_url, prefix)
        # Replace newlines with literal \n so capture_logprobs.py's
        # one-prompt-per-line loader still works.
        single_line = text_pref.replace("\\", "\\\\").replace("\n", "\\n")
        out = args.out_dir / f"long_corpus_{L}tok.txt"
        out.write_text(single_line + "\n")
        # round-trip sanity: re-tokenize the saved single-line, see how close
        # we are to L (newline encoding may shift a token or two).
        roundtrip = tokenize(args.base_url, text_pref)
        print(f"  {L:5d}-tok prefix -> {out}  ({len(text_pref):,} chars; round-trip {len(roundtrip)} tokens)")


if __name__ == "__main__":
    main()

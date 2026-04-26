#!/usr/bin/env python3
"""Summarize a PyTorch profiler trace by op name.

Reads a chrome-tracing JSON (gzipped or plain) and aggregates total time
per operator. Useful for answering "where does decode time go?".

Usage:
  scripts/summarize-profile.py path/to/trace.json [--top N]
"""

import argparse
import collections
import gzip
import json
import sys
from pathlib import Path


def load_trace(path: Path):
    if path.suffix == ".gz" or path.name.endswith(".json.gz"):
        with gzip.open(path, "rt") as f:
            return json.load(f)
    with open(path) as f:
        return json.load(f)


def categorize(name: str) -> str:
    """Group ops into coarse buckets so we get a high-level breakdown."""
    n = name.lower()
    if any(x in n for x in ("attention", "flash_attn", "sdpa", "paged")):
        return "attention"
    if any(x in n for x in ("rms_norm", "layer_norm", "rmsnorm", "layernorm")):
        return "norm"
    if any(x in n for x in ("rope", "rotary")):
        return "rope"
    if any(x in n for x in ("silu", "swiglu", "geglu", "gelu", "relu")):
        return "activation"
    if any(x in n for x in ("matmul", "linear", "gemm", "addmm", "mm")):
        return "matmul/linear"
    if "moe" in n or "expert" in n:
        return "moe"
    if "sample" in n or "logits" in n or "softmax" in n:
        return "sampling"
    if "kv_cache" in n or "kvcache" in n or "cache" in n:
        return "kv_cache"
    if any(x in n for x in ("memcpy", "copy_", "alloc", "free")):
        return "memory"
    if "embedding" in n:
        return "embedding"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--by", choices=["op", "category"], default="category")
    args = ap.parse_args()

    if not args.trace.exists():
        sys.exit(f"trace not found: {args.trace}")

    print(f"Loading {args.trace} ({args.trace.stat().st_size / 1e6:.1f} MB)...", file=sys.stderr)
    trace = load_trace(args.trace)
    events = trace.get("traceEvents", trace if isinstance(trace, list) else [])

    by_op = collections.defaultdict(lambda: [0, 0])      # name -> [total_us, calls]
    by_cat = collections.defaultdict(lambda: [0, 0])
    total_us = 0
    gpu_events = 0
    cpu_events = 0
    kinds = collections.Counter()

    for ev in events:
        if ev.get("ph") != "X":
            continue
        name = ev.get("name", "?")
        dur = ev.get("dur", 0)
        if not dur:
            continue
        cat = ev.get("cat", "")
        kinds[cat] += 1
        # Only count actual GPU (kernel) events for the summary if present.
        # Otherwise fall back to CPU events.
        if "kernel" in cat or "gpu" in cat.lower():
            gpu_events += 1
            by_op[name][0] += dur
            by_op[name][1] += 1
            by_cat[categorize(name)][0] += dur
            by_cat[categorize(name)][1] += 1
            total_us += dur
        elif cat == "cpu_op" or "cpu" in cat:
            cpu_events += 1

    if total_us == 0:
        # No GPU events captured — fall back to top CPU ops
        print("No GPU/kernel events found. Falling back to CPU op summary.", file=sys.stderr)
        for ev in events:
            if ev.get("ph") != "X":
                continue
            cat = ev.get("cat", "")
            if cat != "cpu_op":
                continue
            name = ev.get("name", "?")
            dur = ev.get("dur", 0)
            by_op[name][0] += dur
            by_op[name][1] += 1
            by_cat[categorize(name)][0] += dur
            by_cat[categorize(name)][1] += 1
            total_us += dur

    print(f"\nEvent kinds:")
    for k, c in kinds.most_common(10):
        print(f"  {k:20s} {c}")
    print(f"\nGPU/kernel events: {gpu_events}, CPU op events: {cpu_events}")
    print(f"Total accounted time: {total_us / 1e3:.1f} ms\n")

    if total_us == 0:
        print("No events with non-zero duration. Trace may be empty.")
        return

    print(f"=== By category ({args.by} bucket) ===")
    target = by_cat if args.by == "category" else by_op
    rows = sorted(target.items(), key=lambda kv: -kv[1][0])[: args.top]
    for name, (us, n) in rows:
        pct = 100 * us / total_us
        print(f"  {pct:5.1f}%  {us/1e3:9.2f} ms  {n:8d} calls   {name}")


if __name__ == "__main__":
    main()

"""Benchmark our flash attention against baseline vllm-xpu-kernels attention.

Run inside the vllm-intel-arc container or with vLLM XPU installed:
  python benchmarks/bench_against_baseline.py

Compares:
  1. vLLM's built-in XPU attention (baseline)
  2. Our custom xpu-flash-attention kernel
  3. PyTorch SDPA on XPU (reference)
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class BenchResult:
    name: str
    batch: int
    heads: int
    seq_len: int
    head_dim: int
    latency_ms: float
    tflops: float
    bandwidth_gbs: float


def benchmark_pytorch_sdpa(batch, heads, seq_len, head_dim, device, warmup=5, iters=20):
    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)

    for _ in range(warmup):
        torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        if device.type == "xpu":
            torch.xpu.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        if device.type == "xpu":
            torch.xpu.synchronize()
    elapsed = (time.perf_counter() - start) / iters * 1000

    flops = 4.0 * batch * heads * seq_len * seq_len * head_dim
    tflops = flops / (elapsed * 1e9)
    bytes_moved = 4 * batch * heads * seq_len * head_dim * 2  # fp16
    bw = bytes_moved / (elapsed * 1e6)

    return BenchResult("pytorch_sdpa", batch, heads, seq_len, head_dim, elapsed, tflops, bw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="xpu", choices=["xpu", "cpu", "cuda"])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.device == "xpu":
        import intel_extension_for_pytorch  # noqa: F401

    configs = [
        (1, 32, 512, 128),
        (1, 32, 2048, 128),
        (1, 32, 4096, 128),
        (1, 32, 8192, 128),
        (4, 32, 2048, 128),
        (1, 8, 2048, 128),   # GQA
        (1, 8, 8192, 128),
    ]

    results = []

    print(f"Device: {device}")
    if args.device == "xpu":
        print(f"GPU: {torch.xpu.get_device_name(0)}")
    print()

    print("=== PyTorch SDPA (baseline) ===")
    for b, h, s, d in configs:
        r = benchmark_pytorch_sdpa(b, h, s, d, device)
        results.append(r)
        print(f"  B={b} H={h} S={s} D={d} | {r.latency_ms:.2f} ms | {r.tflops:.2f} TFLOPS | {r.bandwidth_gbs:.1f} GB/s")

    # TODO: Add benchmark for our custom kernel once bindings are built
    # print("\n=== XPU Flash Attention (ours) ===")
    # try:
    #     import xpu_flash_attn
    #     ...

    # TODO: Add benchmark for vLLM's built-in attention
    # print("\n=== vLLM XPU Attention (built-in) ===")
    # try:
    #     from vllm._C import ops
    #     ...

    if args.output:
        with open(args.output, "w") as f:
            json.dump([r.__dict__ for r in results], f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

"""Benchmark suite for vLLM on Intel Arc Pro B70.

Measures throughput and latency across different:
- Prompt lengths (128, 512, 1024, 2048, 4096)
- Output lengths (128, 256, 512)
- Concurrency levels (1, 4, 8, 16, 32)
- Configurations (baseline, flash-attn, kvcache-quant)

Output: JSON results file for analysis and comparison.
"""

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import aiohttp


@dataclass
class BenchmarkResult:
    config: str
    prompt_len: int
    output_len: int
    concurrency: int
    total_tokens: int
    elapsed_s: float
    throughput_tps: float
    avg_latency_ms: float
    p50_latency_ms: float
    p99_latency_ms: float
    ttft_ms: float  # time to first token


async def generate_one(session, url, model, prompt, max_tokens):
    """Send a single completion request and measure timing."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }

    start = time.perf_counter()
    first_token_time = None
    total_tokens = 0

    async with session.post(f"{url}/v1/completions", json=payload) as resp:
        async for line in resp.content:
            line = line.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                if chunk.get("choices", [{}])[0].get("text"):
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    total_tokens += 1
            except json.JSONDecodeError:
                continue

    elapsed = time.perf_counter() - start
    ttft = (first_token_time - start) if first_token_time else elapsed

    return total_tokens, elapsed, ttft


async def run_concurrent(url, model, prompt, max_tokens, concurrency):
    """Run concurrent requests and aggregate results."""
    async with aiohttp.ClientSession() as session:
        tasks = [
            generate_one(session, url, model, prompt, max_tokens)
            for _ in range(concurrency)
        ]
        results = await asyncio.gather(*tasks)

    tokens = [r[0] for r in results]
    latencies = [r[1] * 1000 for r in results]  # ms
    ttfts = [r[2] * 1000 for r in results]

    total_tokens = sum(tokens)
    total_elapsed = max(r[1] for r in results)

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p99 = latencies[int(len(latencies) * 0.99)]

    return BenchmarkResult(
        config="baseline",
        prompt_len=len(prompt.split()),
        output_len=max_tokens,
        concurrency=concurrency,
        total_tokens=total_tokens,
        elapsed_s=total_elapsed,
        throughput_tps=total_tokens / total_elapsed if total_elapsed > 0 else 0,
        avg_latency_ms=sum(latencies) / len(latencies),
        p50_latency_ms=p50,
        p99_latency_ms=p99,
        ttft_ms=sum(ttfts) / len(ttfts),
    )


def make_prompt(target_tokens):
    """Generate a prompt of approximately target_tokens length."""
    # ~1.3 words per token on average
    word = "The quick brown fox jumps over the lazy dog. "
    words_needed = int(target_tokens * 1.3)
    repetitions = max(1, words_needed // 9)
    return (word * repetitions).strip()


async def main_async(args):
    prompt_lens = args.prompt_lens
    output_lens = args.output_lens
    concurrencies = args.concurrencies

    results = []

    print(f"Model: {args.model}")
    print(f"Prompt lengths: {prompt_lens}")
    print(f"Output lengths: {output_lens}")
    print(f"Concurrency levels: {concurrencies}")
    print()

    for prompt_len in prompt_lens:
        prompt = make_prompt(prompt_len)
        for output_len in output_lens:
            for conc in concurrencies:
                print(f"  P={prompt_len} O={output_len} C={conc} ... ", end="", flush=True)
                try:
                    result = await run_concurrent(
                        args.base_url, args.model, prompt, output_len, conc
                    )
                    result.prompt_len = prompt_len
                    results.append(result)
                    print(f"{result.throughput_tps:.1f} tok/s, "
                          f"TTFT={result.ttft_ms:.0f}ms, "
                          f"p50={result.p50_latency_ms:.0f}ms")
                except Exception as e:
                    print(f"FAILED: {e}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\nResults saved to {args.output}")

    return results


def main():
    parser = argparse.ArgumentParser(description="vLLM Intel Arc Benchmark")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--concurrencies",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[1, 4, 8, 16],
        help="Comma-separated list of concurrency levels (e.g. 1,8,16,32)",
    )
    parser.add_argument(
        "--prompt-lens",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[128, 512, 1024, 2048],
        help="Comma-separated prompt token targets",
    )
    parser.add_argument(
        "--output-lens",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[128, 256],
        help="Comma-separated output token targets",
    )
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

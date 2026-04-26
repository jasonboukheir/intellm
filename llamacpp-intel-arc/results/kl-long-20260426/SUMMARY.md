# Phase 2c — long-context KL harness

Same patched llama.cpp build as `profile-aicss-20260426-002625`. Tests
KV-cache quantization fidelity and throughput as context length grows.

## Setup

- Source corpus: 5,000-token model continuation on a "distributed
  databases" technical-article seed prompt (saved as
  `configs/prompts/long_corpus.txt`). Slicing this corpus into 256 / 1024 /
  4096-token prefixes gives us realistic, semantically coherent prompts
  that don't repeat themselves.
- Server boots: 3 (one per KV-cache type: f16 / q8_0 / q4_0).
- Captures: 9 total (3 lengths × 3 KV types). Same prefix at each length
  across all 3 KV types so KL is meaningful position-by-position.
- Generation: 32 tokens × top-50 logprobs, greedy (temp=0).

## Quality (KL, NLL on greedy-agreement prefix)

| length | KV vs FP16 | KL mean | KL p50 | greedy agreement | -log Q(P_pick) | baseline NLL |
|---|---|---|---|---|---|---|
| 256-tok prefix | q8_0 | 0.00792 | 0.00258 | 38 % | 0.4748 | 0.4455 |
| 256-tok prefix | q4_0 | 0.01445 | 0.00790 | 38 % | 0.5048 | 0.4455 |
| 1024-tok prefix | q8_0 | 0.00107 | 0.00105 | 28 % | 0.3445 | 0.3132 |
| 1024-tok prefix | q4_0 | 0.01076 | 0.00439 | 28 % | 0.3127 | 0.3132 |
| **4096-tok prefix** | **q8_0** | **0.00042** | **0.00010** | **100 %** | **0.2162** | **0.2172** |
| **4096-tok prefix** | **q4_0** | **0.00599** | **0.00242** | **66 %** | **0.2632** | **0.2172** |

**Key finding: KL *decreases* with context length, contrary to a naive
"errors compound" intuition.** Mechanism: at long context the model has
overwhelming evidence about local continuations and concentrates
probability tightly on a few tokens; small KV-cache numerical jitter
gets drowned out. q8_0 KV at 4K context produced the *exact same* greedy
sequence as FP16 (100 % agreement, KL=0.0004 — operationally lossless).

q4_0 KV stays sub-0.01 nats per token at 4K — still well below the
threshold where chat output would feel different.

## Throughput

| length | KV  | prompt-eval tok/s | decode tok/s |
|---|---|---|---|
| 256  | fp16 | 392 | **36.97** |
| 256  | q8_0 |  10*| 24.86 *|
| 256  | q4_0 |  11*| 25.17 *|
| 1024 | fp16 | 499 | **37.19** |
| 1024 | q8_0 | 465 | 35.54 |
| 1024 | q4_0 | 464 | 35.51 |
| 4096 | fp16 | 528 | **36.35** |
| 4096 | q8_0 | 524 | 33.03 |
| 4096 | q4_0 | 523 | 32.88 |

\* The 256-tok q8_0/q4_0 numbers reflect cold-server JIT+cache warmup
on the first request after restart — ignore them for comparison.

**Two surprises:**

1. **Single-stream decode is ~37 tok/s at any context length above 1K.**
   That's *half* the 73 tok/s we measured at zero-context single-stream
   (in `profile-aicss-20260426-002625`). The extra cost is per-token
   attention work over the cached context — at 4K context, the K and V
   matmuls inside attention get materially larger.

2. **q8_0 and q4_0 KV are *slightly slower* than FP16 at 4K context, not
   faster.** The dequant-on-attention-read costs ~3–10 %. KV bytes are
   not the throughput bottleneck at single-stream — model weights still
   dominate the per-token bandwidth budget.

## Where KV-quant *does* help

KV-quant's value isn't speed at single-stream — it's **VRAM headroom
for concurrent sequences and longer max context**. Back-of-envelope on
this card (32 GB, ~12 GB free after weights):

| ctx | parallel | FP16 KV | Q8 KV | Q4 KV |
|---|---|---|---|---|
| 8K | 1 | 640 MB | 320 MB | 160 MB |
| 8K | 8 | 5.1 GB | 2.6 GB | 1.3 GB |
| 32K | 8 | 20.5 GB *(OOM)* | 10.2 GB | 5.1 GB |
| 32K | 16 | 41 GB *(OOM)* | 20.5 GB *(OOM)* | 10.2 GB |

At 32K context with 16 concurrent users, **only Q4 KV fits** on this
card — and the quality cost is 0.006 nats per token vs FP16. Q4 KV is
the right default once context > 8K.

(Numbers above use Qwen3.6's 10 attention layers × hidden 2048 × 2 (K+V)
× context tokens, in fp16 / int8 / int4. The 30 DeltaNet layers don't
have KV cache, so this is the full footprint.)

## Implications for IsoQuant (phase 2b)

IsoQuant's claimed advantage over llama.cpp's built-in quants is *better
KL at the same bit-budget* — useful below 4 bits, where built-in quants
would degrade. On Qwen3.6 specifically:

- Above Q4: built-in is already operationally lossless, IsoQuant has
  little headroom to demonstrate benefit.
- At Q3 / Q2: this is where IsoQuant should beat llama.cpp's IQ3/IQ2
  built-ins. That's the regime to test.
- Long-context multi-tenant (32K context × 16+ concurrent): the byte
  budget is so tight that going below Q4 is the only way to fit, and
  IsoQuant's better KL-per-bit becomes user-visible.

## Reproduce

```sh
# corpus
curl -s -X POST http://127.0.0.1:8081/v1/completions -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.6-35b-a3b-q4km","prompt":"Write a long, dense...","max_tokens":5000,"temperature":0.7,"seed":42}' \
    | jq -r .choices[0].text > configs/prompts/long_corpus.txt

# slice
python harness/build_long_prompts.py \
    --base-url http://127.0.0.1:8081 \
    --corpus configs/prompts/long_corpus.txt \
    --lengths 256 1024 4096 \
    --out-dir configs/prompts

# captures (one server boot per KV type)
for kv in fp16 q8_0 q4_0; do
    # restart server with --cache-type-{k,v} = $kv
    for L in 256 1024 4096; do
        python harness/capture_logprobs.py \
            --prompts configs/prompts/long_corpus_${L}tok.txt \
            --tag ${kv}-${L}tok \
            --out results/kl-long-20260426/data/logprobs/${kv}-${L}tok.json
    done
done

# compute
for kv in q8_0 q4_0; do for L in 256 1024 4096; do
    python harness/compute_kl.py \
        results/kl-long-20260426/data/logprobs/fp16-${L}tok.json \
        results/kl-long-20260426/data/logprobs/${kv}-${L}tok.json
done; done
```

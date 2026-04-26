# v1 baseline (Battlemage / Arc B70, 0xe223)

Device info: 256 max compute units, 128 KB SLM, max work-group 1024.

## bench_flash_attn (v1, FP16, causal=true)

| B | H  | S    | D   | latency (ms) | TFLOPS | GB/s |
|---|----|------|-----|--------------|--------|------|
| 1 | 32 |  512 | 128 |    5.77      |  0.74  |  2.9 |
| 1 | 32 | 2048 | 128 |   56.16      |  1.22  |  1.2 |
| 1 | 32 | 4096 | 128 |  190.02      |  1.45  |  0.7 |
| 1 | 32 | 8192 | 128 |  703.34      |  1.56  |  0.4 |
| 4 | 32 | 2048 | 128 |  184.04      |  1.49  |  1.5 |
| 1 |  8 | 2048 | 128 |   22.80      |  0.75  |  0.7 |
| 1 |  8 | 8192 | 128 |  196.58      |  1.40  |  0.3 |

## PyTorch SDPA on XPU (torch 2.10 native, FP16, causal=true)

| B | H  | S    | D   | latency (ms) | TFLOPS | GB/s  |
|---|----|------|-----|--------------|--------|-------|
| 1 | 32 |  512 | 128 |     0.35     |  12.10 |  47.3 |
| 1 | 32 | 2048 | 128 |     0.66     | 104.56 | 102.1 |
| 1 | 32 | 4096 | 128 |     1.74     | 158.27 |  77.3 |
| 1 | 32 | 8192 | 128 |     6.89     | 159.61 |  39.0 |
| 4 | 32 | 2048 | 128 |     2.10     | 131.14 | 128.1 |
| 1 |  8 | 2048 | 128 |     0.19     |  88.82 |  86.7 |
| 1 |  8 | 8192 | 128 |     1.89     | 145.78 |  35.6 |

## Gap

v1 vs torch SDPA at S=8192: **0.22% (~450x slower).**

Root cause: v1 is doing scalar FMAs in the inner loop. We need:
1. XMX (joint_matrix) for QK^T and PV matmuls — biggest win
2. Sub-group cooperative loading from SLM with vector reads
3. Larger BLOCK_M / BLOCK_N (memory permitting) for arithmetic intensity
4. Software pipelining (load next K tile while computing current)

## Goal

SYCL*TLA reaches ~78% of peak — for Battlemage that's ~160 TFLOPS based on
the torch SDPA numbers (which presumably also use SYCL*TLA underneath).
Our goal: match or beat that.

# llamacpp-intel-arc

llama.cpp + SYCL backend on Intel Arc Pro B70 (Battlemage / Xe2). Picked
over vLLM-XPU for q4 GGUF support — vLLM-XPU only loads FP16 / dynamic FP8 /
MXFP4 (the last has no model checkmarks today). See sibling vllm-intel-arc
profile (`results/profile-20260425-232306/SUMMARY.md`): decode there is
86% of DRAM bandwidth on a 14GB BF16 model. The lever for higher decode
TPS is q4 weights + MoE expert sparsity, both of which llama.cpp+SYCL
delivers for Qwen3.6-35B-A3B (UD-Q4_K_M, 22.1 GB).

## Known Battlemage gotchas

- Compiling with `GGML_SYCL_F16=ON` + `GGML_SYCL_DEVICE_ARCH=bmg_g21`
  causes weight corruption / nonsense output unless `GGML_SYCL_DISABLE_OPT=1`
  is set at runtime.  
  Tracking issue: https://github.com/ggml-org/llama.cpp/issues/21893
- Q8_0 hits only ~21–24% of peak DRAM BW on Xe2 (kernel inefficiency),
  vs Q4_K_M at ~53–64%. Q8_0 generation is ~4× slower than Q4_K_M.  
  Tracking issue: https://github.com/ggml-org/llama.cpp/issues/21517
- vLLM-XPU's FA2 already on Battlemage means custom flash-attention
  upstream contributions belong in vllm-xpu-kernels, not here.

## References

- llama.cpp SYCL: https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md
- Container image: ghcr.io/ggml-org/llama.cpp:server-intel
- Unsloth Qwen3.6 GGUF: https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF

2026-05-07 tick 2: container couldn't see host HF cache
(only kernels + vllm were bind-mounted), so vllm serve would have
re-downloaded the 35B model. Patched `flake.nix` to also bind-mount
`$HOME/.cache/huggingface` → `/root/.cache/huggingface` (override via
`HOST_HF_HOME`); container recreated; verified
`models--palmfuture--Qwen3.6-35B-A3B-GPTQ-Int4` visible from inside
container. Note: container recreate dropped the in-container cmake
cache; host-side `.so` files in
`vllm-xpu-kernels/vllm_xpu_kernels/*.so` survived (bind-mounted),
so `_xpu_C.abi3.so` + `libgdn_attn_kernels_xe_2.so` are still loadable.


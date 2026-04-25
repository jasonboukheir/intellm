# nix-intel-xpu

Nix packaging for Intel XPU/GPU development dependencies not yet in nixpkgs.

## Architecture

- `pkgs/sycl-tla/` — Header-only SYCL*TLA (Templates for Linear Algebra), the CUTLASS equivalent for Intel GPUs
- `pkgs/oneapi-env/` — FHS environment for running Intel oneAPI DPC++ compiler (icpx -fsycl)
- `pkgs/vllm-xpu-kernels/` — Source packaging of vllm-xpu-kernels for reference/development

## Key constraints

- Intel DPC++ compiler (icpx) cannot be cleanly packaged in nix yet — use the `oneapi-env` FHS wrapper
- SYCL*TLA is header-only so it packages trivially
- `level-zero` and `intel-compute-runtime` ARE in nixpkgs — use those directly
- Target hardware: Intel Arc Pro B70 (Xe2/Battlemage, 32GB GDDR6)

## Development workflow

```
nix develop                          # enter dev shell with GPU tools
nix build .#sycl-tla                 # build sycl-tla headers package
nix run .#oneapi-env                 # enter FHS env for DPC++ work
```

## Downstream consumers

- `xpu-flash-attention` — uses sycl-tla headers + oneapi-env for kernel compilation
- `xpu-kvcache-quant` — uses sycl-tla headers + oneapi-env for kernel compilation
- `vllm-intel-arc` — uses all packages for integration

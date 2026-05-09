# intellm — agent guide

This is a meta-repo. Two forks live as submodules: `vllm`, `vllm-xpu-kernels`.

The XPU substrate (torch+xpu, triton-xpu, oneAPI/MKL/SYCL, vllm-xpu-kernels,
vllm, auto-round-xpu, quantize, kl-eval) is provided nix-natively by the
upstream [`vllm-xpu-nix`](https://github.com/jasonboukheir/vllm-xpu-nix)
flake — no containers, no `intel/vllm` image, no host-managed venvs.

## Discover capabilities via the CLI, not this file

```bash
direnv allow                # one-time, pulls dev shell automatically
intellm-help                # listing of meta CLIs and dev shells
intellm-status              # branch / commit / dirty for each submodule
```

The repo `flake.nix` wires the upstream `vllm-xpu-nix` flake onto the local
submodule layout and adds repo-level meta CLIs. Run `--help` on any CLI for
its own usage.

## First-time setup

```bash
git clone --recurse-submodules ssh://git@codeberg.org/jasonboukheir/intellm.git
cd intellm
direnv allow
intellm-init                # if you forgot --recurse-submodules
```

## Submodule conventions

Each submodule tracks one branch (see `.gitmodules`). Branch strategy per
fork:

- One branch per upstream PR — small, clean, rebasable on `main`.
- One integration branch — merges PR branches together for local dev.
  Submodule HEAD points at this.

When an upstream PR lands, delete its branch and rebuild the integration
branch from `main` + remaining branches.

## What lives where

- **In a submodule** (candidate for upstream PR): real source changes,
  regression tests, build/CMake changes.
- **In intellm root** (never upstream): progress journals, scratch probes,
  the meta CLIs (`intellm-status`, `intellm-init`, `intellm-update`,
  `intellm-help`), and the Qwen3.6-35B-A3B preset script
  (`nix/auto-round/auto-round-qwen-3-6-35b-a3b.sh`) with empirical bs/ga
  measurement notes.
- **In `vllm-xpu-nix`** (upstream): all derivations and dev shells —
  `auto-round-xpu`, `quantize`, `kl-eval`, `vllm-xpu`, `vllm-xpu-kernels`,
  the `kernels-dev` / `vllm-dev` / `attn-dev` shells.

## Iterating against local submodule checkouts

```bash
# vllm-xpu-kernels: full toolchain + closure, then editable install
nix develop .#kernels-dev
cd vllm-xpu-kernels
pip install -e . --no-build-isolation

# vllm: full toolchain + closure, including a vllm-xpu-kernels build
nix develop .#vllm-dev
cd vllm
pip install -e . --no-build-isolation --no-deps

# fast in-tree iteration on attn_kernels_xe_2
nix develop .#attn-dev
make dev-attn KERNELS_SRC=$PWD/vllm-xpu-kernels
```

To build the `unstable` variants of the upstream packages from the local
submodules (instead of upstream's pinned source):

```bash
nix build .#vllm-xpu-kernels-unstable \
  --override-input vllm-xpu-nix/vllm-xpu-kernels-unstable-src path:./vllm-xpu-kernels
nix build .#vllm-xpu-unstable \
  --override-input vllm-xpu-nix/vllm-xpu-unstable-src path:./vllm
```

## Domain-specific guides

`vllm/AGENTS.md` is upstream contributor guidance — read it before touching
the vllm submodule. `vllm-xpu-kernels`' upstream README is the source of
truth for kernel work. The `vllm-xpu-nix` repo has its own `docs/`
covering build, NixOS overlay use, hardware prerequisites, and the
quantize/eval workflow.

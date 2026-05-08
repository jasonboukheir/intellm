# intellm — agent guide

This is a meta-repo. Three forks live as submodules:
`vllm`, `vllm-xpu-kernels`, `auto-round`.

## Discover capabilities via the CLI, not this file

```bash
direnv allow                # one-time, pulls dev shell automatically
intellm-help                # all CLIs across submodules
intellm-status              # branch / commit / dirty for each submodule
```

All CLIs are defined in the meta-repo's `flake.nix`. The submodules
themselves carry no nix or dev infra — they stay clean for upstream PRs.
Run any CLI with `--help` for its own usage.

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
  the dev-ergonomics flake, CLI wrappers, and per-fork nix infra:
  - `nix/auto-round/` — Containerfile + podman wrappers for the XPU
    quantization toolkit (autoround, auto-round-qwen-3-6-35b-a3b).
  - `flake.nix` — also defines the Docker-based vllm-xpu-kernels and vllm
    CLIs (vllm-xpu-build, vllm-test, etc.).
  - `scripts/` — `quantize.sh`, `kl_eval.py` (run inside the auto-round
    container).

## Domain-specific guides

`vllm/AGENTS.md` is upstream contributor guidance — read it before touching
the vllm submodule. The other two submodules carry no agent files; their
upstream READMEs are the source of truth.

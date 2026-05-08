# intellm — agent guide

This is a meta-repo. Three forks live as submodules:
`vllm`, `vllm-xpu-kernels`, `auto-round`.

## Discover capabilities via the CLI, not this file

```bash
direnv allow                # one-time, pulls dev shell automatically
intellm-help                # all CLIs across submodules
intellm-status              # branch / commit / dirty for each submodule
```

Each forwarded CLI runs `nix develop ./<submodule> -c <cli>` — pick up the
same env you'd get from `cd <submodule>; nix develop`. Run any CLI with
`--help` for its own usage.

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
  dev-ergonomics flake, CLI wrappers.

## Domain-specific guides

Each submodule has its own `AGENTS.md` / `CLAUDE.md`. Read those before
making non-trivial changes inside a submodule.

{
  description = "intellm — meta-repo for vllm, vllm-xpu-kernels forks. Wires the upstream vllm-xpu-nix flake (nix-native XPU substrate) onto local submodule checkouts and adds repo-level meta CLIs.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    vllm-xpu-nix = {
      url = "github:jasonboukheir/vllm-xpu-nix";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };
  };

  outputs = { self, nixpkgs, flake-utils, vllm-xpu-nix }:
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        upstream = vllm-xpu-nix.packages.${system};
        upstreamShells = vllm-xpu-nix.devShells.${system};

        submodules = [ "vllm" "vllm-xpu-kernels" ];

        intellmStatus = pkgs.writeShellApplication {
          name = "intellm-status";
          runtimeInputs = [ pkgs.git ];
          text = ''
            root="$(git rev-parse --show-toplevel)"
            cd "$root"
            printf "%-20s %-30s %-10s %s\n" "submodule" "branch" "commit" "dirty?"
            printf "%-20s %-30s %-10s %s\n" "---------" "------" "------" "------"
            ${pkgs.lib.concatMapStringsSep "\n" (s: ''
              if [ -d "${s}/.git" ] || [ -f "${s}/.git" ]; then
                cd "$root/${s}"
                branch="$(git branch --show-current 2>/dev/null || echo 'detached')"
                commit="$(git rev-parse --short HEAD 2>/dev/null)"
                dirty="$(git status --porcelain 2>/dev/null | wc -l)"
                if [ "$dirty" -gt 0 ]; then dirty="$dirty file(s)"; else dirty="clean"; fi
                printf "%-20s %-30s %-10s %s\n" "${s}" "$branch" "$commit" "$dirty"
                cd "$root"
              else
                printf "%-20s %s\n" "${s}" "(not initialized — run intellm-init)"
              fi
            '') submodules}
          '';
        };

        intellmInit = pkgs.writeShellApplication {
          name = "intellm-init";
          runtimeInputs = [ pkgs.git ];
          text = ''
            root="$(git rev-parse --show-toplevel)"
            cd "$root"
            git submodule update --init --recursive --jobs 4
            echo "intellm: submodules initialized."
            echo "Tracking branches:"
            git config -f .gitmodules --get-regexp 'submodule\..*\.branch'
          '';
        };

        intellmUpdate = pkgs.writeShellApplication {
          name = "intellm-update";
          runtimeInputs = [ pkgs.git ];
          text = ''
            root="$(git rev-parse --show-toplevel)"
            cd "$root"
            git submodule update --remote --jobs 4
            git status -s -- vllm vllm-xpu-kernels
            echo ""
            echo "If submodule pins changed above, commit the bumps:"
            echo "  git add <submodule>"
            echo "  git commit -m 'bump <submodule> to latest <branch>'"
          '';
        };

        # Pinned ruff binaries — match each subproject's .pre-commit-config.yaml
        # so `lint` produces the same diagnostics CI would. Nixpkgs' ruff drifts
        # ahead of vLLM's pin (jasonbk memory note: 0.15.x vs 0.14.0), and
        # `uv tool run` trips on uv's bundled glibc python under NixOS, so we
        # vendor the official musl static binary per version.
        mkRuff = { version, sha256 }: pkgs.stdenvNoCC.mkDerivation {
          pname = "ruff";
          inherit version;
          src = pkgs.fetchurl {
            url = "https://github.com/astral-sh/ruff/releases/download/${version}/ruff-x86_64-unknown-linux-musl.tar.gz";
            inherit sha256;
          };
          sourceRoot = "ruff-x86_64-unknown-linux-musl";
          dontConfigure = true;
          dontBuild = true;
          installPhase = ''
            install -Dm755 ruff "$out/bin/ruff"
          '';
        };

        ruffVllm    = mkRuff { version = "0.14.0"; sha256 = "sha256-7W0bhAeh0ijcMy+xkFfobgSmzTwr6s2zJK1v8qP5Bxs="; };
        ruffKernels = mkRuff { version = "0.11.7"; sha256 = "sha256-DK8yqww5m/ugjRixkNwarsjqRzaIZKS/Oz8RPjbo/Wg="; };

        intellmLint = pkgs.writeShellApplication {
          name = "lint";
          runtimeInputs = [ pkgs.git ];
          text = ''
            root="$(git rev-parse --show-toplevel)"
            cd "$root"

            run_in() {
              local sub="$1" ruff="$2"
              if [ ! -d "$sub" ]; then
                echo "[$sub] not initialized — skipping"
                return 0
              fi
              echo "=== $sub ==="
              (
                cd "$sub"
                if [ -f .pre-commit-config.yaml ] && command -v pre-commit >/dev/null 2>&1; then
                  echo "[$sub] pre-commit run --all-files"
                  pre-commit run --all-files
                else
                  if [ ! -f .pre-commit-config.yaml ]; then
                    echo "[$sub] no .pre-commit-config.yaml — running ruff only"
                  else
                    echo "[$sub] pre-commit not installed — falling back to ruff only"
                  fi
                  echo "[$sub] ruff check ."
                  "$ruff" check .
                  echo "[$sub] ruff format --check ."
                  "$ruff" format --check .
                fi
              )
            }

            fail=0
            run_in vllm              "${ruffVllm}/bin/ruff"    || fail=1
            run_in vllm-xpu-kernels  "${ruffKernels}/bin/ruff" || fail=1
            exit "$fail"
          '';
        };

        intellmHelp = pkgs.writeShellApplication {
          name = "intellm-help";
          text = ''
            cat <<'EOF'
            intellm — meta-repo for vllm, vllm-xpu-kernels.

            All XPU substrate (torch+xpu, triton-xpu, oneAPI/MKL/SYCL, vllm-xpu-
            kernels, vllm) comes from the upstream vllm-xpu-nix flake — no
            containers. This repo adds submodule wiring and per-repo ergonomics
            on top.

            Meta CLIs:
              intellm-status            show branch/commit/dirty for each submodule
              intellm-init              initialize submodules (first-time setup)
              intellm-update            pull latest tracked branch for each submodule
              lint                      run pre-commit (or pinned ruff fallback) across both subprojects
              intellm-help              this message

            vllm + vllm-xpu-kernels (iterate against the local submodules):
              nix develop .#kernels-dev          toolchain + closure for vllm-xpu-kernels
                                                 cd vllm-xpu-kernels && pip install -e . --no-build-isolation
              nix develop .#vllm-dev             toolchain + closure for vllm
                                                 cd vllm && pip install -e . --no-build-isolation --no-deps
              nix develop .#attn-dev             fast in-tree iteration on attn_kernels_xe_2

            Pre-built packages (also exposed via overlay; see vllm-xpu-nix README):
              nix build .#vllm-xpu               vllm pinned to upstream vllm-project
              nix build .#vllm-xpu-unstable      vllm pinned to jasonboukheir fork
              nix build .#vllm-xpu-kernels       upstream-pinned kernels
              nix build .#vllm-xpu-kernels-unstable
              nix build .#torch-xpu / .#triton-xpu

            Build the unstable variants against your local submodule:
              nix build .#vllm-xpu-kernels-unstable \
                --override-input vllm-xpu-nix/vllm-xpu-kernels-unstable-src path:./vllm-xpu-kernels
              nix build .#vllm-xpu-unstable \
                --override-input vllm-xpu-nix/vllm-xpu-unstable-src path:./vllm
            EOF
          '';
        };

        metaCommands = [ intellmStatus intellmInit intellmUpdate intellmLint intellmHelp ];

        # Re-export upstream packages so users can `nix build .#vllm-xpu` etc.
        # without remembering the flake URL.
        upstreamPackages = {
          inherit (upstream)
            intel-oneapi intel-pti oneccl-bmg
            torch-xpu triton-xpu
            flash-linear-attention
            vllm-xpu-kernels vllm-xpu-kernels-unstable
            vllm-xpu vllm-xpu-unstable;
        };

      in {
        devShells.default = pkgs.mkShell {
          name = "intellm-dev";
          packages = metaCommands ++ [
            pkgs.git
            pkgs.gnumake
            pkgs.jujutsu
            pkgs.gh
            pkgs.direnv
            pkgs.pre-commit
            pkgs.level-zero
            pkgs.intel-compute-runtime
            pkgs.clinfo
            pkgs.pciutils
          ];
          shellHook = ''
            export INTELLM_ROOT="$(git rev-parse --show-toplevel)"
            echo "intellm dev shell. Run 'intellm-help' for the full CLI listing."
          '';
        };

        # Re-export upstream's iteration shells so users can stay anchored on
        # the intellm flake (and its submodule layout) without juggling URLs.
        devShells.kernels-dev = upstreamShells.kernels-dev;
        devShells.vllm-dev    = upstreamShells.vllm-dev;
        devShells.attn-dev    = upstreamShells.attn-dev;

        packages = upstreamPackages // (builtins.listToAttrs (map (p: { name = p.name; value = p; }) metaCommands));

        apps.default = {
          type = "app";
          program = "${intellmHelp}/bin/intellm-help";
        };
      });
}

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

        metaCommands = [ intellmStatus intellmInit intellmUpdate intellmHelp ];

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

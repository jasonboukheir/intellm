{
  description = "intellm — meta-repo for vllm, vllm-xpu-kernels, auto-round forks. Wires the upstream vllm-xpu-nix flake (nix-native XPU substrate) onto local submodule checkouts and adds repo-level meta CLIs.";

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

        # auto-round-xpu env exposes `auto-round`, `auto-round-light`,
        # `auto-round-best` as native entry points — no container.
        autoRoundEnv = pkgs.python312Packages.python.withPackages (_: [ upstream.auto-round-xpu ]);

        # Repo-specific Qwen3.6-35B-A3B preset (empirical bs/ga tuning notes
        # in the script). Wraps the auto-round-xpu env directly.
        qwenPresetScript = ./nix/auto-round/auto-round-qwen-3-6-35b-a3b.sh;
        autoroundQwen35b = pkgs.writeShellApplication {
          name = "auto-round-qwen-3-6-35b-a3b";
          runtimeInputs = [ autoRoundEnv pkgs.git ];
          text = ''
            if [ -z "''${AUTOROUND_OUTPUT_DIR:-}" ]; then
              root="''${INTELLM_ROOT:-$(git rev-parse --show-superproject-working-tree 2>/dev/null || git rev-parse --show-toplevel 2>/dev/null || pwd)}"
              export AUTOROUND_OUTPUT_DIR="$root/output/auto-round"
            fi
            exec bash "${qwenPresetScript}" "$@"
          '';
        };

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
            git status -s -- vllm vllm-xpu-kernels auto-round
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
            intellm — meta-repo for vllm, vllm-xpu-kernels, auto-round.

            All XPU substrate (torch+xpu, triton-xpu, oneAPI/MKL/SYCL, vllm-xpu-
            kernels, vllm, auto-round-xpu) comes from the upstream vllm-xpu-nix
            flake — no containers. This repo adds submodule wiring and per-repo
            ergonomics on top.

            Meta CLIs:
              intellm-status            show branch/commit/dirty for each submodule
              intellm-init              initialize submodules (first-time setup)
              intellm-update            pull latest tracked branch for each submodule
              intellm-help              this message

            Quantization (nix-native, no container):
              quantize <model> <type>            B70-tuned AutoRound wrapper.
                                                 types: int4 int8 mxfp4 nvfp4 gguf:q4_k_m ...
                                                 Recipes via AUTOROUND_QUANTIZE_RECIPE env:
                                                   default   (200i, ~4.4h)
                                                   light     (50i, ~1.7h)
                                                   overnight (400i + patience 100, ~8-14h)
                                                   best      (1000i, days).
                                                 Run 'quantize help' for full options.
              kl-eval [args]                     KL/top-1 eval of a quantized model vs its
                                                 BF16 reference. --quant-model points at the
                                                 directory `quantize` writes to.
              auto-round-qwen-3-6-35b-a3b safe        bs=4 ga=2 + drop low_gpu_mem (~5h, ~29 GB est)
              auto-round-qwen-3-6-35b-a3b aggressive  bs=8 ga=1 + drop low_gpu_mem (~4h, tight)
              auto-round-qwen-3-6-35b-a3b help        full preset listing

              auto-round / auto-round-light / auto-round-best are also on PATH
              if you want to bypass the wrapper.

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
              nix build .#torch-xpu / .#triton-xpu / .#auto-round-xpu

            Build the unstable variants against your local submodule:
              nix build .#vllm-xpu-kernels-unstable \
                --override-input vllm-xpu-nix/vllm-xpu-kernels-unstable-src path:./vllm-xpu-kernels
              nix build .#vllm-xpu-unstable \
                --override-input vllm-xpu-nix/vllm-xpu-unstable-src path:./vllm
            EOF
          '';
        };

        metaCommands = [ intellmStatus intellmInit intellmUpdate intellmHelp autoroundQwen35b ];

        # Re-export upstream packages so users can `nix build .#vllm-xpu` etc.
        # without remembering the flake URL.
        upstreamPackages = {
          inherit (upstream)
            intel-oneapi intel-pti oneccl-bmg
            torch-xpu triton-xpu
            flash-linear-attention
            auto-round-xpu
            vllm-xpu-kernels vllm-xpu-kernels-unstable
            vllm-xpu vllm-xpu-unstable
            quantize kl-eval;
        };

      in {
        devShells.default = pkgs.mkShell {
          name = "intellm-dev";
          packages = metaCommands ++ [
            autoRoundEnv
            upstream.quantize
            upstream.kl-eval
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
            export AUTOROUND_OUTPUT_DIR="''${AUTOROUND_OUTPUT_DIR:-$INTELLM_ROOT/output/auto-round}"
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

        apps.quantize = vllm-xpu-nix.apps.${system}.quantize;
        apps.kl-eval  = vllm-xpu-nix.apps.${system}.kl-eval;
        apps.autoround = vllm-xpu-nix.apps.${system}.autoround;
      });
}

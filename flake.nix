{
  description = "intellm — meta-repo for vllm, vllm-xpu-kernels, auto-round forks";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        submodules = [ "vllm" "vllm-xpu-kernels" "auto-round" ];

        # Forward a CLI to the corresponding submodule's nix dev shell.
        # `nix develop ./<submodule> -c <cli> "$@"` enters the subrepo flake's
        # default shell and runs the CLI defined there.
        mkForward = { name, submodule }: pkgs.writeShellApplication {
          inherit name;
          runtimeInputs = [ pkgs.nix pkgs.git ];
          text = ''
            root="''${INTELLM_ROOT:-$(git rev-parse --show-superproject-working-tree 2>/dev/null || git rev-parse --show-toplevel 2>/dev/null || pwd)}"
            target="$root/${submodule}"
            if [ ! -f "$target/flake.nix" ]; then
              echo "intellm: ${submodule} flake missing at $target — did you run 'git submodule update --init'?" >&2
              exit 1
            fi
            exec nix develop "$target" --command ${name} "$@"
          '';
        };

        # CLIs that vllm-xpu-kernels' flake exposes inside its dev shell.
        # Source of truth: vllm-xpu-kernels/flake.nix (lines 67-203).
        xpuCommands = [
          "vllm-xpu-build" "vllm-xpu-rebuild" "vllm-xpu-test" "vllm-xpu-bench"
          "vllm-xpu-shell" "vllm-xpu-status" "vllm-xpu-clean"
          "vllm-test" "vllm-shell" "vllm-run"
        ];
        xpuForwards = map (n: mkForward { name = n; submodule = "vllm-xpu-kernels"; }) xpuCommands;

        # CLIs that auto-round's flake exposes.
        # Source of truth: auto-round/flake.nix.
        autoroundCommands = [
          "autoround" "auto-round-qwen-3-6-35b-a3b" "quantize" "commands"
        ];
        autoroundForwards = map (n: mkForward { name = n; submodule = "auto-round"; }) autoroundCommands;

        # Meta CLIs: operate across submodules.
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

        # KL-divergence eval: AutoRound-quantized model vs BF16 reference.
        # Lives in intellm root (dev-ergonomics) and runs scripts/kl_eval.py
        # inside the auto-round XPU container (built once by `autoround build`).
        # Avoids touching the auto-round submodule.
        #
        # Path conventions inside the container:
        #   /output            <- intellm/auto-round/output (where quantize writes)
        #   /intellm-scripts   <- intellm/scripts (this script)
        #   /workspace         <- intellm/auto-round (live source overlay)
        #   /root/.cache/intellm/kl-eval  <- host ~/.cache/intellm/kl-eval (logp cache)
        klEval = pkgs.writeShellApplication {
          name = "kl-eval";
          runtimeInputs = [ pkgs.podman pkgs.git ];
          text = ''
            root="''${INTELLM_ROOT:-$(git rev-parse --show-superproject-working-tree 2>/dev/null || git rev-parse --show-toplevel 2>/dev/null || pwd)}"
            script="$root/scripts/kl_eval.py"
            image="''${AUTOROUND_IMAGE_TAG:-localhost/auto-round-xpu:latest}"
            if [ ! -f "$script" ]; then
              echo "kl-eval: $script missing" >&2
              exit 1
            fi
            if ! podman image exists "$image"; then
              echo "kl-eval: image '$image' not found." >&2
              echo "  build it once with:  nix develop ./auto-round -c autoround build" >&2
              exit 1
            fi
            hf_cache="''${HF_HOME:-$HOME/.cache/huggingface}"
            output_base="''${AUTOROUND_OUTPUT_DIR:-$root/auto-round/output}"
            kl_cache="$HOME/.cache/intellm"
            mkdir -p "$hf_cache" "$output_base" "$kl_cache/kl-eval"
            tty_args=()
            if [ -t 0 ] && [ -t 1 ]; then
              tty_args=(-it)
            fi
            exec podman run "''${tty_args[@]}" --rm \
              --device /dev/dri \
              --group-add keep-groups \
              --shm-size=16g \
              -v "$hf_cache:/root/.cache/huggingface:z" \
              -v "$output_base:/output:z" \
              -v "$root/scripts:/intellm-scripts:z,ro" \
              -v "$kl_cache:/root/.cache/intellm:z" \
              -v "$root/auto-round:/workspace:z" \
              -e PYTHONPATH=/workspace \
              "$image" \
              python /intellm-scripts/kl_eval.py "$@"
          '';
        };

        intellmHelp = pkgs.writeShellApplication {
          name = "intellm-help";
          text = ''
            cat <<'EOF'
            intellm — meta-repo for vllm, vllm-xpu-kernels, auto-round.

            Meta CLIs (operate across submodules):
              intellm-status            show branch/commit/dirty for each submodule
              intellm-init              initialize submodules (first-time setup)
              intellm-update            pull latest tracked branch for each submodule
              intellm-help              this message
              kl-eval [args]            KL/top-1 eval of a quantized model vs its BF16 reference
                                        (runs scripts/kl_eval.py inside the auto-round container).
                                        --quant-model expects a path under /output (the host
                                        auto-round/output dir is bind-mounted there).

            vllm-xpu-kernels CLIs (forward into ./vllm-xpu-kernels nix shell):
              vllm-xpu-build            full kernel build (Debug; CMAKE_BUILD_TYPE=Release for perf)
              vllm-xpu-rebuild          incremental ninja install after edits
              vllm-xpu-test [args]      pytest in vllm-xpu-kernels
              vllm-xpu-bench script     run a benchmark script with PYTHONPATH=.
              vllm-xpu-shell            interactive bash inside dev container
              vllm-xpu-status           container state
              vllm-xpu-clean            remove dev container (keeps source bind mount)

            vllm CLIs (forward into ./vllm-xpu-kernels nix shell, run vs ./vllm):
              vllm-test [args]          pytest in vllm
              vllm-shell                interactive bash in vllm working tree
              vllm-run [-e K=V] cmd     run arbitrary command against vllm

            auto-round CLIs (forward into ./auto-round nix shell):
              autoround                 the auto-round CLI
              auto-round-qwen-3-6-35b-a3b   pre-canned Qwen 3.6 35B-A3B quantize
              quantize                  generic quantize wrapper
              commands                  list available auto-round subcommands

            Each forwarded CLI runs `nix develop ./<submodule> -c <cli>` —
            i.e. it picks up the same env as `cd <submodule>; nix develop`.
            Run any CLI with --help (when supported) for its own usage.
            EOF
          '';
        };

        metaCommands = [ intellmStatus intellmInit intellmUpdate intellmHelp klEval ];
        allWrappers = metaCommands ++ xpuForwards ++ autoroundForwards;

      in {
        devShells.default = pkgs.mkShell {
          name = "intellm-dev";
          packages = allWrappers ++ [
            pkgs.git
            pkgs.gnumake
            pkgs.jujutsu
            pkgs.gh
            pkgs.direnv
          ];
          shellHook = ''
            echo "intellm dev shell. Run 'intellm-help' for commands."
            export INTELLM_ROOT="$(git rev-parse --show-toplevel)"
          '';
        };

        # Expose the meta CLIs as packages too, so `nix run .#intellm-status` works.
        packages = builtins.listToAttrs (map (p: { name = p.name; value = p; }) allWrappers);

        # Default app is intellm-help.
        apps.default = {
          type = "app";
          program = "${intellmHelp}/bin/intellm-help";
        };
      });
}

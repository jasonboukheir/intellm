{
  description = "intellm — meta-repo for vllm, vllm-xpu-kernels, auto-round forks (dev shell + nix infra for all three)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        submodules = [ "vllm" "vllm-xpu-kernels" "auto-round" ];

        # ─────────────────────────────────────────────────────────────────────
        # auto-round — podman-wrapped XPU quantization toolkit.
        # Scripts live at ./nix/auto-round/, kept in this meta-repo so the
        # auto-round submodule stays clean for upstream PRs.
        # ─────────────────────────────────────────────────────────────────────
        autoroundContainerfile = ./nix/auto-round/Containerfile;
        autoroundBuildContext  = ./nix/auto-round;
        autoroundScript        = ./nix/auto-round/autoround.sh;
        autoroundQwen35bScript = ./nix/auto-round/auto-round-qwen-3-6-35b-a3b.sh;

        autoroundRuntimeInputs = [
          pkgs.podman
          pkgs.skopeo
          pkgs.jq
          pkgs.curl
          pkgs.gawk
        ];

        autoround = pkgs.writeShellApplication {
          name = "autoround";
          runtimeInputs = autoroundRuntimeInputs;
          text = ''
            export AUTOROUND_CONTAINERFILE="${autoroundContainerfile}"
            export AUTOROUND_BUILD_CONTEXT="${autoroundBuildContext}"
            # Default the live-source overlay to the auto-round submodule in
            # this meta-repo. Override with AUTOROUND_SOURCE_DIR=/path/to/repo
            # to point at a different checkout.
            if [ -z "''${AUTOROUND_SOURCE_DIR:-}" ]; then
              root="''${INTELLM_ROOT:-$(git rev-parse --show-superproject-working-tree 2>/dev/null || git rev-parse --show-toplevel 2>/dev/null || pwd)}"
              if [ -d "$root/auto-round/auto_round" ]; then
                export AUTOROUND_SOURCE_DIR="$root/auto-round"
              fi
            fi
            if [ -z "''${AUTOROUND_OUTPUT_DIR:-}" ]; then
              root="''${INTELLM_ROOT:-$(git rev-parse --show-superproject-working-tree 2>/dev/null || git rev-parse --show-toplevel 2>/dev/null || pwd)}"
              export AUTOROUND_OUTPUT_DIR="$root/output/auto-round"
            fi
            exec bash "${autoroundScript}" "$@"
          '';
        };

        autoroundQwen35b = pkgs.writeShellApplication {
          name = "auto-round-qwen-3-6-35b-a3b";
          runtimeInputs = [ autoround ];
          text = ''exec bash "${autoroundQwen35bScript}" "$@"'';
        };

        # ─────────────────────────────────────────────────────────────────────
        # vllm-xpu-kernels — Docker-based build/test loop for vllm + kernels.
        # Bind-mounts the kernels checkout (this submodule) and the vllm
        # checkout (sibling submodule) into a single long-lived container.
        # ─────────────────────────────────────────────────────────────────────
        vllmDevImage     = "localhost/vllm-xpu-int4-tq:gdn-fix-ccd77bdf4-squashed";
        vllmDevContainer = "vllm-dev";
        kernelsMount     = "/workspace/vllm-xpu-kernels";
        vllmMount        = "/workspace/vllm";

        ensureContainer = ''
          IMAGE="''${VLLM_DEV_IMAGE:-''${VLLM_XPU_KERNELS_IMAGE:-${vllmDevImage}}}"
          ROOT="''${INTELLM_ROOT:-$(git rev-parse --show-superproject-working-tree 2>/dev/null || git rev-parse --show-toplevel 2>/dev/null || pwd)}"
          KERNELS_ROOT="''${VLLM_XPU_KERNELS_REPO_ROOT:-$ROOT/vllm-xpu-kernels}"
          VLLM_ROOT="''${VLLM_REPO_ROOT:-$ROOT/vllm}"
          if [ ! -d "$KERNELS_ROOT" ]; then
            echo "warning: vllm-xpu-kernels not at $KERNELS_ROOT — run intellm-init?" >&2
          fi
          if [ ! -d "$VLLM_ROOT" ]; then
            echo "warning: VLLM_REPO_ROOT=$VLLM_ROOT does not exist; vllm will not be mounted." >&2
            VLLM_MOUNT_ARGS=""
          else
            VLLM_MOUNT_ARGS="-v $VLLM_ROOT:${vllmMount}"
          fi
          HOST_HF_DIR="''${HOST_HF_HOME:-$HOME/.cache/huggingface}"
          if [ -d "$HOST_HF_DIR" ]; then
            HF_MOUNT_ARGS="-v $HOST_HF_DIR:/root/.cache/huggingface"
          else
            HF_MOUNT_ARGS=""
          fi
          if ! docker ps --filter "name=^/${vllmDevContainer}$" --format '{{.Names}}' | grep -q '^${vllmDevContainer}$'; then
            if docker ps -a --filter "name=^/${vllmDevContainer}$" --format '{{.Names}}' | grep -q '^${vllmDevContainer}$'; then
              docker start ${vllmDevContainer} >/dev/null
            else
              echo "Creating container ${vllmDevContainer} from $IMAGE..."
              # shellcheck disable=SC2086
              docker run -d --name ${vllmDevContainer} \
                --device /dev/dri \
                --ipc=host \
                -v "$KERNELS_ROOT:${kernelsMount}" \
                $VLLM_MOUNT_ARGS \
                $HF_MOUNT_ARGS \
                -w ${kernelsMount} \
                --entrypoint sleep \
                "$IMAGE" infinity >/dev/null
              echo "Pinning setuptools to <80 (one-time)..."
              docker exec ${vllmDevContainer} pip install -q 'setuptools>=77.0.3,<80.0.0'
              if [ -n "$VLLM_MOUNT_ARGS" ]; then
                echo "Replacing baked-in vllm with editable install from ${vllmMount} (one-time)..."
                docker exec ${vllmDevContainer} sh -c \
                  'pip uninstall -y vllm >/dev/null 2>&1 || true; \
                   pip install -e ${vllmMount} --no-deps --no-build-isolation -q'
              fi
            fi
          fi
        '';

        mkVllmScript = name: text: pkgs.writeShellScriptBin name ''
          set -eo pipefail
          ${ensureContainer}
          ${text}
        '';

        forwardVllmEnv = ''
          ENV_PASSTHROUGH=""
          while IFS='=' read -r k _; do
            case "$k" in
              VLLM_*|TORCH_*|CMAKE_*|MAX_JOBS|PYTHONPATH|HF_*|HUGGINGFACE_*)
                ENV_PASSTHROUGH="$ENV_PASSTHROUGH -e $k" ;;
            esac
          done < <(env)
        '';

        # MAX_JOBS auto-tunes by feature set: SYCL-TLA chunk kernels in
        # gdn_attn/xe_2 and grouped_gemm/xe_2 use ~40 GB RAM each in icpx,
        # so two heavies on a 96 GiB box OOM-kill the compiler. When those
        # targets are in, default to 2; otherwise 6.
        vllmXpuBuild = mkVllmScript "vllm-xpu-build" ''
          gdn_on=1; gemm_on=1
          [ "''${GDN_KERNELS_ENABLED:-ON}" = "OFF" ] && gdn_on=0
          [ "''${MOE_KERNELS_ENABLED:-ON}" = "OFF" ] && gemm_on=0
          if [ "$gdn_on$gemm_on" = "00" ]; then
            export MAX_JOBS="''${MAX_JOBS:-6}"
          else
            export MAX_JOBS="''${MAX_JOBS:-2}"
          fi
          export VLLM_XPU_AOT_DEVICES="''${VLLM_XPU_AOT_DEVICES:-bmg}"
          export VLLM_XPU_XE2_AOT_DEVICES="''${VLLM_XPU_XE2_AOT_DEVICES:-bmg}"
          export CMAKE_BUILD_TYPE="''${CMAKE_BUILD_TYPE:-Debug}"
          ENV_PASSTHROUGH=""
          for var in MAX_JOBS CMAKE_BUILD_TYPE \
                     VLLM_XPU_AOT_DEVICES VLLM_XPU_XE2_AOT_DEVICES \
                     BASIC_KERNELS_ENABLED FA2_KERNELS_ENABLED \
                     MOE_KERNELS_ENABLED GDN_KERNELS_ENABLED \
                     XPU_SPECIFIC_KERNELS_ENABLED XPUMEM_ALLOCATOR_ENABLED \
                     VLLM_XPU_ENABLE_XE2 VLLM_XPU_ENABLE_XE_DEFAULT; do
            if [ -n "''${!var+x}" ]; then
              ENV_PASSTHROUGH="$ENV_PASSTHROUGH -e $var"
            fi
          done
          echo "Full build (CMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE, AOT=$VLLM_XPU_AOT_DEVICES, MAX_JOBS=$MAX_JOBS)..."
          echo "  Per-session override: GDN_KERNELS_ENABLED=OFF vllm-xpu-build (etc.)"
          # shellcheck disable=SC2086
          exec docker exec --workdir ${kernelsMount} \
            $ENV_PASSTHROUGH \
            ${vllmDevContainer} \
            sh -c 'rm -rf build && pip install -e . --no-build-isolation'
        '';

        vllmXpuRebuild = mkVllmScript "vllm-xpu-rebuild" ''
          # ninja install lays the abi3 .so files in build/temp/install/, but
          # the runtime-shared libs (libgdn_attn_kernels_xe_2.so etc.) live one
          # level up at build/temp/. The pip-editable layout imports from
          # vllm_xpu_kernels/ in the source tree, so copy both groups so
          # reimports pick up the new build.
          exec docker exec --workdir ${kernelsMount} ${vllmDevContainer} sh -c \
            'cd build/temp && ninja install \
               && cp -f install/vllm_xpu_kernels/*.so ../../vllm_xpu_kernels/ \
               && cp -f lib*_xe_*.so ../../vllm_xpu_kernels/ 2>/dev/null || true'
        '';

        vllmXpuTest = mkVllmScript "vllm-xpu-test" ''
          ${forwardVllmEnv}
          # shellcheck disable=SC2086
          exec docker exec --workdir ${kernelsMount} $ENV_PASSTHROUGH ${vllmDevContainer} pytest "$@"
        '';

        vllmXpuBench = mkVllmScript "vllm-xpu-bench" ''
          ${forwardVllmEnv}
          # shellcheck disable=SC2086
          exec docker exec --workdir ${kernelsMount} -e PYTHONPATH=. $ENV_PASSTHROUGH ${vllmDevContainer} python "$@"
        '';

        vllmXpuShell = mkVllmScript "vllm-xpu-shell" ''
          ${forwardVllmEnv}
          # shellcheck disable=SC2086
          exec docker exec -it --workdir ${kernelsMount} $ENV_PASSTHROUGH ${vllmDevContainer} bash
        '';

        vllmTest = mkVllmScript "vllm-test" ''
          ${forwardVllmEnv}
          # shellcheck disable=SC2086
          exec docker exec --workdir ${vllmMount} $ENV_PASSTHROUGH ${vllmDevContainer} pytest "$@"
        '';

        vllmShell = mkVllmScript "vllm-shell" ''
          ${forwardVllmEnv}
          # shellcheck disable=SC2086
          exec docker exec -it --workdir ${vllmMount} $ENV_PASSTHROUGH ${vllmDevContainer} bash
        '';

        vllmRun = mkVllmScript "vllm-run" ''
          ${forwardVllmEnv}
          EXPLICIT_ENV=""
          while [ "$#" -gt 0 ] && [ "$1" = "-e" ]; do
            EXPLICIT_ENV="$EXPLICIT_ENV -e $2"
            shift 2
          done
          # shellcheck disable=SC2086
          exec docker exec --workdir ${vllmMount} $ENV_PASSTHROUGH $EXPLICIT_ENV ${vllmDevContainer} "$@"
        '';

        vllmXpuStatus = pkgs.writeShellScriptBin "vllm-xpu-status" ''
          docker ps -a --filter "name=^/${vllmDevContainer}$" --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
        '';

        vllmXpuClean = pkgs.writeShellScriptBin "vllm-xpu-clean" ''
          docker rm -f ${vllmDevContainer} 2>/dev/null || true
          docker rm -f swiglu-dev 2>/dev/null || true
          echo "Removed ${vllmDevContainer} (and legacy swiglu-dev if present)."
          echo "Next build/test will recreate ${vllmDevContainer} with the current flake's mounts."
        '';

        # ─────────────────────────────────────────────────────────────────────
        # Meta CLIs: operate across submodules.
        # ─────────────────────────────────────────────────────────────────────
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

        # B70-tuned AutoRound wrapper. Runs scripts/quantize.sh inside the
        # auto-round XPU container (same image `autoround` builds).
        quantize = pkgs.writeShellApplication {
          name = "quantize";
          runtimeInputs = [ autoround pkgs.git ];
          text = ''
            root="''${INTELLM_ROOT:-$(git rev-parse --show-superproject-working-tree 2>/dev/null || git rev-parse --show-toplevel 2>/dev/null || pwd)}"
            script="$root/scripts/quantize.sh"
            if [ ! -f "$script" ]; then
              echo "intellm: $script missing" >&2
              exit 1
            fi
            exec bash "$script" "$@"
          '';
        };

        # KL-divergence eval: AutoRound-quantized model vs BF16 reference.
        # Runs scripts/kl_eval.py inside the auto-round XPU container (built
        # once by `autoround build`).
        #
        # Path conventions inside the container:
        #   /output            <- intellm/output/auto-round (where quantize writes)
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
              echo "  build it once with:  autoround build" >&2
              exit 1
            fi
            hf_cache="''${HF_HOME:-$HOME/.cache/huggingface}"
            output_base="''${AUTOROUND_OUTPUT_DIR:-$root/output/auto-round}"
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
            All nix/dev infra for the three forks lives in this repo; the
            submodules themselves stay clean for upstream PRs.

            Meta CLIs:
              intellm-status            show branch/commit/dirty for each submodule
              intellm-init              initialize submodules (first-time setup)
              intellm-update            pull latest tracked branch for each submodule
              intellm-help              this message

            auto-round CLIs (podman, XPU container):
              autoround <model>                       quantize with sensible defaults
                                                      (W4A16 auto-round-light, low_gpu_mem ON)
              autoround quantize <model> [flags...]   explicit form
              autoround shell                         interactive shell inside the container
              autoround run -- <cmd...>               arbitrary command inside the container
              autoround build                         rebuild image (--no-cache)
              autoround pull                          pull base image only
              auto-round-qwen-3-6-35b-a3b safe        bs=4 ga=2 + drop low_gpu_mem (~5h, ~29 GB)
              auto-round-qwen-3-6-35b-a3b aggressive  bs=8 ga=1 + drop low_gpu_mem (~4h, tight)
              auto-round-qwen-3-6-35b-a3b help        full preset listing

              quantize <model> <type>                 B70-tuned wrapper (runs scripts/quantize.sh
                                                      inside the auto-round container).
                                                      types: int4 int8 mxfp4 nvfp4 gguf:q4_k_m ...
                                                      Recipes via AUTOROUND_QUANTIZE_RECIPE env:
                                                        default   (200i, ~4.4h)
                                                        light     (50i, ~1.7h)
                                                        overnight (400i + patience 100, ~8-14h)
                                                        best      (1000i, days).
                                                      Run 'quantize help' for full options.
              kl-eval [args]                          KL/top-1 eval of a quantized model vs its
                                                      BF16 reference (runs scripts/kl_eval.py
                                                      inside the auto-round container).
                                                      --quant-model paths are under /output.

            vllm-xpu-kernels CLIs (Docker, vllm-dev container; lazy-created on first build/test):
              vllm-xpu-build              full kernel build (Debug; CMAKE_BUILD_TYPE=Release for perf;
                                          per-session feature flags: GDN_KERNELS_ENABLED=OFF etc.)
              vllm-xpu-rebuild            incremental ninja install after edits
              vllm-xpu-test [pytest args] pytest in vllm-xpu-kernels
              vllm-xpu-bench script       run a benchmark script with PYTHONPATH=.
              vllm-xpu-shell              interactive bash in /workspace/vllm-xpu-kernels
              vllm-xpu-status             container state
              vllm-xpu-clean              remove dev container (keeps source bind mount)

            vllm CLIs (run vs the vllm submodule via the same vllm-dev container):
              vllm-test [pytest args]     pytest in vllm
              vllm-shell                  interactive bash in /workspace/vllm
              vllm-run [-e K=V ...] cmd   exec arbitrary command in /workspace/vllm

            Override base image:   VLLM_DEV_IMAGE=...  (default: localhost/vllm-xpu-int4-tq:gdn-fix-ccd77bdf4-squashed)
            Override vllm root:    VLLM_REPO_ROOT=...  (default: <intellm>/vllm)
            Override kernels root: VLLM_XPU_KERNELS_REPO_ROOT=... (default: <intellm>/vllm-xpu-kernels)
            EOF
          '';
        };

        autoroundCommands = [ autoround autoroundQwen35b ];
        vllmKernelsCommands = [
          vllmXpuBuild vllmXpuRebuild vllmXpuTest vllmXpuBench vllmXpuShell
          vllmTest vllmShell vllmRun
          vllmXpuStatus vllmXpuClean
        ];
        metaCommands = [ intellmStatus intellmInit intellmUpdate intellmHelp quantize klEval ];
        allWrappers = metaCommands ++ autoroundCommands ++ vllmKernelsCommands;

      in {
        devShells.default = pkgs.mkShell {
          name = "intellm-dev";
          packages = allWrappers ++ autoroundRuntimeInputs ++ [
            pkgs.git
            pkgs.gnumake
            pkgs.jujutsu
            pkgs.gh
            pkgs.direnv
            pkgs.dive
            pkgs.level-zero
            pkgs.intel-compute-runtime
            pkgs.clinfo
            pkgs.pciutils
          ];
          shellHook = ''
            export INTELLM_ROOT="$(git rev-parse --show-toplevel)"
            export AUTOROUND_SOURCE_DIR="''${AUTOROUND_SOURCE_DIR:-$INTELLM_ROOT/auto-round}"
            export AUTOROUND_OUTPUT_DIR="''${AUTOROUND_OUTPUT_DIR:-$INTELLM_ROOT/output/auto-round}"
            echo "intellm dev shell. Run 'intellm-help' for the full CLI listing."
          '';
        };

        packages = builtins.listToAttrs (map (p: { name = p.name; value = p; }) allWrappers);

        apps.default = {
          type = "app";
          program = "${intellmHelp}/bin/intellm-help";
        };

        checks.smoke = pkgs.runCommand "intellm-smoke"
          { nativeBuildInputs = [ pkgs.bash pkgs.shellcheck ]; }
          ''
            echo "--- bash -n (auto-round scripts) ---"
            bash -n ${autoroundScript}
            bash -n ${autoroundQwen35bScript}
            echo "--- shellcheck (auto-round scripts) ---"
            shellcheck ${autoroundScript} || true
            shellcheck ${autoroundQwen35bScript} || true
            touch $out
          '';
      });
}

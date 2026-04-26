{
  description = "vLLM integration for Intel Arc Pro GPUs — benchmarking, containers, and custom kernel integration";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    nix-intel-xpu.url = "path:../nix-intel-xpu";
  };

  outputs = { self, nixpkgs, flake-utils, nix-intel-xpu }:
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        # intel/vllm:0.17.0-xpu pinned by content digest. Refresh with:
        #   skopeo inspect docker://intel/vllm:0.17.0-xpu --format '{{.Digest}}'
        vllmImage = "intel/vllm@sha256:e961d08135a6a8ef6decd857c6deab7a70eb00e19de21de54cbc0ce05d9a9f43";

        python = pkgs.python312.withPackages (ps: with ps; [
          numpy
          torch
          pytest
          matplotlib
          pandas
          pyyaml
          requests
          aiohttp
          tqdm
          transformers
          huggingface-hub
          datasets
          scipy
          sentencepiece
        ]);

        runtimeInputs = [
          python
          pkgs.podman
          pkgs.skopeo
          pkgs.jq
          pkgs.yq-go
          pkgs.curl
        ];

        # Wraps a project script so its tooling comes from Nix and the container
        # image digest is pinned, regardless of caller PATH/env.
        mkApp = name: script: flake-utils.lib.mkApp {
          drv = pkgs.writeShellApplication {
            name = "vllm-intel-arc-${name}";
            runtimeInputs = runtimeInputs;
            text = ''
              if [ ! -f "scripts/${script}" ] || [ ! -f flake.nix ]; then
                echo "error: run this from the vllm-intel-arc project root (no scripts/${script} in $PWD)" >&2
                exit 1
              fi
              export VLLM_IMAGE="${vllmImage}"
              exec bash "scripts/${script}" "$@"
            '';
          };
        };

      in {
        devShells.default = pkgs.mkShell {
          name = "vllm-intel-arc-dev";
          packages = runtimeInputs ++ [
            pkgs.dive
            pkgs.level-zero
            pkgs.intel-compute-runtime
            pkgs.pciutils
            pkgs.clinfo
            pkgs.httpie
          ];

          VLLM_IMAGE = vllmImage;

          shellHook = ''
            echo "vllm-intel-arc development environment"
            echo ""
            echo "One-shot entrypoints:"
            echo "  nix run .#baseline       # pull (if needed) → server → throughput → quality"
            echo "  nix run .#server         # start vLLM server only"
            echo "  nix run .#benchmark      # throughput against running server"
            echo "  nix run .#quality-eval -- <tag>  # capture logprobs"
            echo "  nix run .#pull-container # pull pinned image"
            echo "  nix flake check          # syntax + config sanity (no GPU)"
            echo ""
            echo "Image: $VLLM_IMAGE"
          '';
        };

        devShells.oneapi = nix-intel-xpu.packages.${system}.oneapi-env;

        apps = {
          baseline = mkApp "baseline" "full-baseline.sh";
          server = mkApp "server" "run-server.sh";
          benchmark = mkApp "benchmark" "run-benchmark.sh";
          quality-eval = mkApp "quality-eval" "run-quality-eval.sh";
          pull-container = mkApp "pull-container" "pull-container.sh";
          default = mkApp "baseline" "full-baseline.sh";
        };

        checks.smoke = pkgs.runCommand "vllm-intel-arc-smoke"
          {
            nativeBuildInputs = [ pkgs.bash pkgs.yq-go ];
          }
          ''
            cd ${self}
            echo "--- bash -n ---"
            for f in scripts/*.sh; do
              echo "  $f"
              bash -n "$f"
            done
            echo "--- yaml configs ---"
            for f in configs/models/*.yaml; do
              echo "  $f"
              yq -e '.model' "$f" >/dev/null
            done
            touch $out
          '';
      }
    );
}

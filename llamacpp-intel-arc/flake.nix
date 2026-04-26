{
  description = "llama.cpp + SYCL on Intel Arc Pro B70 (Battlemage) — q4 GGUF inference, benchmarks, KL-div harness";

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

        # ggml-org official Intel/SYCL build of llama.cpp's server.
        # Pin by digest once we've smoke-tested a tag — for now, tag-pinned.
        llamaImage = "ghcr.io/ggml-org/llama.cpp:server-intel";

        python = pkgs.python312.withPackages (ps: with ps; [
          numpy
          scipy
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
          sentencepiece
        ]);

        runtimeInputs = [
          python
          pkgs.podman
          pkgs.skopeo
          pkgs.jq
          pkgs.yq-go
          pkgs.curl
          pkgs.wget
        ];

        mkApp = name: script: flake-utils.lib.mkApp {
          drv = pkgs.writeShellApplication {
            name = "llamacpp-intel-arc-${name}";
            runtimeInputs = runtimeInputs;
            text = ''
              if [ ! -f "scripts/${script}" ] || [ ! -f flake.nix ]; then
                echo "error: run from llamacpp-intel-arc project root (no scripts/${script} in $PWD)" >&2
                exit 1
              fi
              export LLAMA_IMAGE="${llamaImage}"
              exec bash "scripts/${script}" "$@"
            '';
          };
        };

      in {
        devShells.default = pkgs.mkShell {
          name = "llamacpp-intel-arc-dev";
          packages = runtimeInputs ++ [
            pkgs.dive
            pkgs.level-zero
            pkgs.intel-compute-runtime
            pkgs.pciutils
            pkgs.clinfo
            pkgs.httpie
          ];

          LLAMA_IMAGE = llamaImage;

          shellHook = ''
            echo "llamacpp-intel-arc development environment"
            echo ""
            echo "One-shot entrypoints:"
            echo "  nix run .#pull-container # pull pinned llama.cpp:server-intel"
            echo "  nix run .#pull-model -- <model.yaml>   # download GGUF"
            echo "  nix run .#server -- <model.yaml>       # llama-server"
            echo "  nix run .#benchmark -- [args]          # throughput against running server"
            echo "  nix run .#profile -- <model.yaml>      # decode profile (workload + metrics)"
            echo "  nix flake check          # syntax + config sanity"
            echo ""
            echo "Image: $LLAMA_IMAGE"
          '';
        };

        apps = {
          pull-container = mkApp "pull-container" "pull-container.sh";
          pull-model     = mkApp "pull-model"     "pull-model.sh";
          server         = mkApp "server"         "run-server.sh";
          benchmark      = mkApp "benchmark"      "run-benchmark.sh";
          profile        = mkApp "profile"        "profile-decode.sh";
          default        = mkApp "server"         "run-server.sh";
        };

        checks.smoke = pkgs.runCommand "llamacpp-intel-arc-smoke"
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
              yq -e '.gguf_file' "$f" >/dev/null
            done
            touch $out
          '';
      }
    );
}

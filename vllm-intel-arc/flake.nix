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

      in {
        devShells.default = pkgs.mkShell {
          name = "vllm-intel-arc-dev";
          packages = [
            python

            # container tooling
            pkgs.podman
            pkgs.skopeo
            pkgs.dive  # inspect container layers

            # gpu tools
            pkgs.level-zero
            pkgs.intel-compute-runtime
            pkgs.pciutils
            pkgs.clinfo

            # utilities
            pkgs.jq
            pkgs.yq-go
            pkgs.curl
            pkgs.httpie
          ];

          VLLM_IMAGE = "intel/vllm:0.17.0-xpu";

          shellHook = ''
            echo "vllm-intel-arc development environment"
            echo ""
            echo "Quick start:"
            echo "  ./scripts/pull-container.sh        # pull intel/vllm container"
            echo "  ./scripts/run-server.sh <model>    # start vLLM server"
            echo "  ./scripts/run-benchmark.sh         # run benchmark suite"
            echo ""
            echo "Container: $VLLM_IMAGE"
          '';
        };

        devShells.oneapi = nix-intel-xpu.packages.${system}.oneapi-env;
      }
    );
}

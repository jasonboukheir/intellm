{
  description = "Nix packaging for Intel XPU/GPU development stack (oneAPI, SYCL*TLA, Level Zero)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    sycl-tla-src = {
      url = "github:intel/sycl-tla";
      flake = false;
    };

    vllm-xpu-kernels-src = {
      url = "github:vllm-project/vllm-xpu-kernels";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, flake-utils, sycl-tla-src, vllm-xpu-kernels-src }:
    {
      overlays.default = final: prev: {
        intel-xpu = {
          sycl-tla = final.callPackage ./pkgs/sycl-tla { src = sycl-tla-src; };
          oneapi-env = final.callPackage ./pkgs/oneapi-env { };
          vllm-xpu-kernels = final.callPackage ./pkgs/vllm-xpu-kernels { src = vllm-xpu-kernels-src; };
        };
      };
    } // flake-utils.lib.eachSystem [ "x86_64-linux" ] (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        sycl-tla = pkgs.callPackage ./pkgs/sycl-tla {
          src = sycl-tla-src;
        };

        oneapi-env = pkgs.callPackage ./pkgs/oneapi-env { };

        vllm-xpu-kernels = pkgs.callPackage ./pkgs/vllm-xpu-kernels {
          src = vllm-xpu-kernels-src;
        };

      in {
        packages = {
          inherit sycl-tla oneapi-env vllm-xpu-kernels;
          default = sycl-tla;
        };

        devShells.default = pkgs.mkShell {
          name = "nix-intel-xpu-dev";
          packages = with pkgs; [
            # build tools
            cmake
            ninja
            pkg-config
            gnumake

            # compilers
            gcc13
            clang_18

            # intel gpu runtime
            level-zero
            intel-compute-runtime
            intel-graphics-compiler

            # python for testing
            (python312.withPackages (ps: with ps; [
              numpy
              pytest
            ]))

            # utilities
            clinfo
            vulkan-tools
            pciutils
          ];

          shellHook = ''
            echo "nix-intel-xpu development environment"
            echo "  level-zero: $(pkg-config --modversion level-zero 2>/dev/null || echo 'not found via pkg-config')"
            echo "  Use 'nix build .#oneapi-env' for FHS environment with oneAPI installer support"
          '';
        };
      }
    );
}

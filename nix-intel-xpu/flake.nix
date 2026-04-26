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

        smoke-test = pkgs.writeShellApplication {
          name = "nix-intel-xpu-smoke-test";
          runtimeInputs = [
            pkgs.bash
            pkgs.clinfo
            pkgs.pciutils
            pkgs.bc
            (pkgs.python312.withPackages (ps: with ps; [ ]))
            pkgs.level-zero
          ];
          text = builtins.readFile ./tests/smoke-test.sh;
        };

      in {
        packages = {
          inherit sycl-tla oneapi-env vllm-xpu-kernels;
          default = sycl-tla;
        };

        apps = {
          smoke-test = flake-utils.lib.mkApp { drv = smoke-test; };
          default = flake-utils.lib.mkApp { drv = smoke-test; };
        };

        checks.smoke-syntax = pkgs.runCommand "nix-intel-xpu-smoke-syntax"
          { nativeBuildInputs = [ pkgs.bash ]; }
          ''
            bash -n ${./tests/smoke-test.sh}
            touch $out
          '';

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
            echo ""
            echo "Entrypoints:"
            echo "  nix run .#smoke-test    # GPU + Level Zero detection"
            echo "  nix run .#oneapi-env    # FHS env for Intel DPC++"
            echo "  nix flake check         # script syntax"
          '';
        };
      }
    );
}

{
  description = "KV cache quantization for Intel XPU — TurboQuant and RotorQuant in SYCL";

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

        intel-xpu = nix-intel-xpu.packages.${system};

        python = pkgs.python312.withPackages (ps: with ps; [
          numpy
          torch
          pytest
          matplotlib
          pandas
          scipy
          pybind11
        ]);

        build = pkgs.writeShellApplication {
          name = "xpu-kvcache-quant-build";
          runtimeInputs = [ pkgs.cmake pkgs.ninja pkgs.pkg-config ];
          text = ''
            if [ ! -f CMakeLists.txt ]; then
              echo "error: run from the xpu-kvcache-quant project root" >&2
              exit 1
            fi
            if ! command -v icpx >/dev/null 2>&1; then
              echo "error: icpx not on PATH — enter the oneAPI FHS env first:" >&2
              echo "  nix run ../nix-intel-xpu#oneapi-env" >&2
              echo "  source /opt/intel/oneapi/setvars.sh" >&2
              exit 1
            fi
            cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release "$@"
            cmake --build build
          '';
        };

      in {
        devShells.default = pkgs.mkShell {
          name = "xpu-kvcache-quant-dev";
          packages = [
            # build
            pkgs.cmake
            pkgs.ninja
            pkgs.pkg-config

            # compilers
            pkgs.gcc13
            pkgs.clang_18

            # intel gpu
            pkgs.level-zero
            pkgs.intel-compute-runtime
            intel-xpu.sycl-tla

            # python
            python

            # debug
            pkgs.gdb
          ];

          CMAKE_PREFIX_PATH = "${intel-xpu.sycl-tla}";

          shellHook = ''
            echo "xpu-kvcache-quant development environment"
            echo ""
            echo "Implements TurboQuant (Google) and RotorQuant (Scrya) for Intel XPU"
            echo ""
            echo "Build (requires oneAPI FHS env first):"
            echo "  nix run ../nix-intel-xpu#oneapi-env"
            echo "  source /opt/intel/oneapi/setvars.sh"
            echo "  nix run .#build"
            echo ""
            echo "Sanity:  nix flake check"
          '';
        };

        devShells.oneapi = nix-intel-xpu.packages.${system}.oneapi-env;

        apps.build = flake-utils.lib.mkApp { drv = build; };
        apps.default = flake-utils.lib.mkApp { drv = build; };

        checks.cmake-configure = pkgs.runCommand "xpu-kvcache-quant-cmake-configure"
          {
            nativeBuildInputs = [ pkgs.cmake pkgs.ninja pkgs.gcc13 ];
          }
          ''
            cp -r ${self}/. project
            chmod -R +w project
            cd project
            # Configure-only sanity check: parses CMakeLists.txt without DPC++ or sycl-tla headers.
            cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
              -DKVQ_SYCL=OFF \
              -DKVQ_BUILD_PYTHON=OFF \
              -DCMAKE_CXX_COMPILER=g++
            touch $out
          '';
      }
    );
}

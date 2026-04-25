{
  description = "Flash attention kernels for Intel Xe2 GPUs (SYCL/DPC++)";

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
          pybind11
        ]);

      in {
        devShells.default = pkgs.mkShell {
          name = "xpu-flash-attention-dev";
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

            # debug/profile
            pkgs.gdb
            pkgs.valgrind
            pkgs.perf-tools
          ];

          CMAKE_PREFIX_PATH = "${intel-xpu.sycl-tla}";

          shellHook = ''
            echo "xpu-flash-attention development environment"
            echo ""
            echo "Build:"
            echo "  cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release"
            echo "  cmake --build build"
            echo ""
            echo "For SYCL compilation, enter oneAPI FHS env:"
            echo "  nix run ../nix-intel-xpu#oneapi-env"
            echo "  source /opt/intel/oneapi/setvars.sh"
          '';
        };

        devShells.oneapi = nix-intel-xpu.packages.${system}.oneapi-env;
      }
    );
}

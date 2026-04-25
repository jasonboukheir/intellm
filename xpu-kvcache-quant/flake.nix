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
            echo "Build:"
            echo "  cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release"
            echo "  cmake --build build"
          '';
        };

        devShells.oneapi = nix-intel-xpu.packages.${system}.oneapi-env;
      }
    );
}

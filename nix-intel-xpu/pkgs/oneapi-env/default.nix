{ lib
, buildFHSEnv
, writeShellScript
, fetchurl
, stdenv
}:

# FHS environment for running Intel oneAPI tools
# oneAPI requires standard Linux filesystem layout that Nix doesn't provide.
# This creates an FHS-compatible sandbox where oneAPI can be installed and run.
#
# Usage:
#   nix run .#oneapi-env
#   # Inside the FHS env:
#   #   Install: ./install-oneapi.sh
#   #   Source:  source /opt/intel/oneapi/setvars.sh
#   #   Compile: icpx -fsycl your_kernel.cpp

let
  oneapi-installer = writeShellScript "install-oneapi.sh" ''
    set -euo pipefail

    ONEAPI_ROOT=/opt/intel/oneapi

    if [ -d "$ONEAPI_ROOT" ]; then
      echo "oneAPI already installed at $ONEAPI_ROOT"
      echo "Source environment: source $ONEAPI_ROOT/setvars.sh"
      exit 0
    fi

    echo "Downloading Intel oneAPI Base Toolkit..."
    echo "Visit: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html"
    echo ""
    echo "For offline installer:"
    echo "  wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/... -O oneapi-installer.sh"
    echo "  chmod +x oneapi-installer.sh"
    echo "  sudo ./oneapi-installer.sh -a --silent --eula accept"
    echo ""
    echo "For apt-based install:"
    echo "  wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | sudo gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg"
    echo "  echo 'deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main' | sudo tee /etc/apt/sources.list.d/oneAPI.list"
    echo "  sudo apt update"
    echo "  sudo apt install intel-oneapi-compiler-dpcpp-cpp intel-oneapi-mkl-devel intel-level-zero-gpu"
  '';

in buildFHSEnv {
  name = "oneapi-env";

  targetPkgs = pkgs: with pkgs; [
    # base development
    gcc13
    clang_18
    cmake
    ninja
    pkg-config
    gnumake

    # intel gpu stack
    level-zero
    intel-compute-runtime
    intel-graphics-compiler

    # required by oneAPI installer / runtime
    coreutils
    bash
    which
    procps
    pciutils
    numactl
    libdrm
    mesa
    zlib
    glib
    xorg.libX11
    xorg.libXext
    xorg.libXrender

    # python
    (python312.withPackages (ps: with ps; [
      pip
      setuptools
      wheel
      numpy
    ]))

    # networking for installer
    wget
    curl
    cacert
    gnupg
  ];

  runScript = "bash";

  extraInstallCommands = ''
    mkdir -p $out/bin
    cp ${oneapi-installer} $out/bin/install-oneapi.sh
  '';

  meta = with lib; {
    description = "FHS environment for Intel oneAPI development";
    platforms = [ "x86_64-linux" ];
  };
}

{ lib
, stdenv
, src
, cmake
, ninja
, python312
, level-zero
}:

# NOTE: Building vllm-xpu-kernels from source requires Intel DPC++ (icpx -fsycl).
# This package currently installs the source tree for reference and development.
# Full compilation requires the oneapi-env FHS environment with DPC++ installed.
#
# For building:
#   nix run .#oneapi-env
#   source /opt/intel/oneapi/setvars.sh
#   cd <this-source>
#   pip install -e .

stdenv.mkDerivation {
  pname = "vllm-xpu-kernels";
  version = "unstable";

  inherit src;

  nativeBuildInputs = [ cmake ninja ];
  buildInputs = [ level-zero python312 ];

  dontBuild = true;

  installPhase = ''
    runHook preInstall
    mkdir -p $out/src
    cp -r . $out/src/vllm-xpu-kernels
    runHook postInstall
  '';

  meta = with lib; {
    description = "XPU (SYCL) kernel implementations for vLLM";
    homepage = "https://github.com/vllm-project/vllm-xpu-kernels";
    platforms = [ "x86_64-linux" ];
  };
}

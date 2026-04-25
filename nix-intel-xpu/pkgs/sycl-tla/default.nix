{ lib
, stdenv
, src
, cmake
, ninja
}:

stdenv.mkDerivation {
  pname = "sycl-tla";
  version = "unstable";

  inherit src;

  nativeBuildInputs = [ cmake ninja ];

  # sycl-tla is header-only — install headers directly
  # The cmake build requires a SYCL compiler, but for packaging we just
  # need the headers available for downstream consumers
  dontBuild = true;

  installPhase = ''
    runHook preInstall
    mkdir -p $out/include
    cp -r $src/include/* $out/include/ 2>/dev/null || true
    cp -r $src/src/* $out/include/ 2>/dev/null || true
    mkdir -p $out/share/sycl-tla
    cp -r $src/examples $out/share/sycl-tla/ 2>/dev/null || true
    runHook postInstall
  '';

  meta = with lib; {
    description = "SYCL Templates for Linear Algebra — CUTLASS equivalent for Intel GPUs";
    homepage = "https://github.com/intel/sycl-tla";
    license = licenses.asl20;
    platforms = [ "x86_64-linux" ];
  };
}

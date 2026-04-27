# Pre-built llama-server (Intel SYCL, Battlemage Arc Pro B70).
#
# We package the binary built outside Nix by `scripts/build-aicss.sh`
# rather than running cmake inside a Nix derivation. SYCL builds need
# Intel oneAPI / icpx, which isn't nix-packaged — and packaging it
# would be ~1-2 weeks of work for closed-source debs + patchelf.
# This is the pattern nixpkgs uses for other unbuildable-from-source
# software (steam, nvidia drivers, discord, etc.).
#
# Each rebuild of the binary produces a new content hash on the input
# path, which gets stamped into the version string via the input's
# narHash. That way every build has a unique
# `llamacpp-intel-arc-server-0.10.0-aicss-<8-hex>` derivation name and
# store path — visible in `nix store ls` history.
{
  stdenvNoCC,
  lib,
  src,
  buildStamp ? "unknown",
}:
  stdenvNoCC.mkDerivation {
    pname = "llamacpp-intel-arc-server";
    version = "0.10.0-aicss-${buildStamp}";

    inherit src;

    dontUnpack = true;
    dontConfigure = true;
    dontBuild = true;
    dontPatchELF = true;
    dontStrip = true;

    installPhase = ''
      runHook preInstall

      mkdir -p $out/bin

      if [ ! -x "$src/llama-server" ]; then
        echo "ERROR: $src/llama-server not found" >&2
        echo "       Build the binary first via:" >&2
        echo "         scripts/build-aicss.sh    # full cherry-pick + build, OR" >&2
        echo "         cmake --build build-aicss/llama-pr-only/build --target llama-server" >&2
        exit 1
      fi

      cp -L "$src/llama-server" "$out/bin/llama-server"

      # Companion shared libs go next to the binary so the runtime
      # container picks them up via LD_LIBRARY_PATH=/llama without the
      # dynamic linker walking back into the Nix store.
      for f in "$src"/lib*.so*; do
        [ -e "$f" ] || continue
        cp -L "$f" "$out/bin/"
      done

      runHook postInstall
    '';

    meta = with lib; {
      description = "Patched llama.cpp server (Intel SYCL, Battlemage)";
      longDescription = ''
        llama.cpp `llama-server` built from upstream master + 6
        cherry-picked Intel SYCL PRs + the IsoQuant rotation patch.
        Designed to run inside the intel/vllm:0.17.0-xpu container
        which supplies oneAPI 2025.3 and level-zero at runtime.
      '';
      homepage = "https://github.com/ggml-org/llama.cpp";
      license = licenses.mit;
      platforms = [ "x86_64-linux" ];
      mainProgram = "llama-server";
    };
  }

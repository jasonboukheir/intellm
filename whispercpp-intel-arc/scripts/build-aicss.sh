#!/usr/bin/env bash
# Build whisper.cpp's whisper-server with the SYCL backend, inside the
# same intel/vllm:0.17.0-xpu runtime image used for the patched
# llama-server build (sibling project ../llamacpp-intel-arc/).
#
# Mainline whisper.cpp — no PR cherry-picks. Whisper's encoder/decoder
# kernels are different from llama's attention path, and the
# Battlemage-targeted SYCL fixes the llamacpp project carries are
# llama-specific. If decode quality regresses on BMG, revisit.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$ROOT/build-aicss/whisper"
RUNTIME_IMAGE="${WHISPER_AICSS_RUNTIME:-docker.io/intel/vllm:0.17.0-xpu}"

if [ ! -d "$BUILD_DIR/.git" ]; then
    echo "[1/2] cloning fresh upstream master into $BUILD_DIR"
    rm -rf "$BUILD_DIR"
    git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git "$BUILD_DIR"
fi

cd "$BUILD_DIR"
git fetch --quiet origin master
git reset --hard origin/master >/dev/null

echo "[2/2] building whisper-server inside $RUNTIME_IMAGE"
# GGML_SYCL_F16 left OFF: sibling llamacpp project's CLAUDE.md notes
# F16 + bmg_g21 corrupts weights without GGML_SYCL_DISABLE_OPT=1 at
# runtime. F32 SYCL is the safe baseline; whisper's compute is small
# enough that F16 perf gain wouldn't matter much anyway.
podman run --rm \
    --device /dev/dri --group-add keep-groups \
    -v "$BUILD_DIR:/work:z" -w /work \
    -e SETVARS_COMPLETED=0 \
    "$RUNTIME_IMAGE" \
    bash -lc '. /opt/intel/oneapi/setvars.sh --force >/dev/null && \
        cmake -S . -B build -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
            -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL \
            -DWHISPER_BUILD_SERVER=ON \
            -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_EXAMPLES=ON \
            -DWHISPER_CURL=OFF -DGGML_CURL=OFF >/dev/null && \
        cmake --build build --target whisper-server -j$(nproc) && \
        cp -L /opt/intel/oneapi/dnnl/*/lib/libdnnl.so.3* build/bin/ && \
        find build -path build/bin -prune -o -name "lib*.so.*.*" -print \
            | xargs -I{} cp -L {} build/bin/'

echo
echo "Done. Binary at $BUILD_DIR/build/bin/whisper-server"
echo "Stage into nix via: ~/.config/nix/scripts/refresh-whispercpp-binary.sh"

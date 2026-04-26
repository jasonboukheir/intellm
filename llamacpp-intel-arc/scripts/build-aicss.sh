#!/usr/bin/env bash
# Build llama.cpp with the open Intel SYCL PRs cherry-picked.
# See build-aicss/README.md for details.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$ROOT/build-aicss/llama-pr-only"
RUNTIME_IMAGE="${LLAMA_AICSS_RUNTIME:-docker.io/intel/vllm:0.17.0-xpu}"

# Cherry-pick targets. Skipping pr-5 because upstream master independently
# added the same dequantize_q8_0_reorder symbols and the cherry-pick produces
# duplicate definitions. See build-aicss/README.md.
declare -A PRS=(
    [22147]=4065b6529304e493236ed7524857f98fa1b970b9
    [22148]=5496728f4593024638ce69c16473ec4c43dc3b73
    [22149]=ad7cabe7f51a6ec1be1f21ddfe2d06805abe3895
    [22150]=1dd517dbb5d797afce03be3352fa04d2a46d219b
    [22153]=210de274941a5e35ee500a0e2453558e409bed64
    [22156]=4f611aec2e645cf87482fc0f455c8ec8f790d7df
)

if [ ! -d "$BUILD_DIR/.git" ]; then
    echo "[1/3] cloning fresh upstream master into $BUILD_DIR"
    rm -rf "$BUILD_DIR"
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$BUILD_DIR"
    cd "$BUILD_DIR"
    git remote add aicss https://github.com/aicss-genai/llama.cpp.git
fi

cd "$BUILD_DIR"

echo "[2/3] cherry-picking 6 PR commits (pr-5 skipped — see README)"
git fetch --quiet aicss "aicss-genai/sycl-bmg-upstream-pr-1" \
                       "aicss-genai/sycl-bmg-upstream-pr-2" \
                       "aicss-genai/sycl-bmg-upstream-pr-3" \
                       "aicss-genai/sycl-bmg-upstream-pr-4" \
                       "aicss-genai/sycl-bmg-upstream-pr-6" \
                       "aicss-genai/sycl-bmg-upstream-pr-7"
git reset --hard origin/master >/dev/null
for n in 22147 22148 22149 22150 22153 22156; do
    sha="${PRS[$n]}"
    if ! git cat-file -e "$sha"^{commit} 2>/dev/null; then
        echo "  missing commit $sha for PR #$n — fetch may not have brought it in"
        exit 1
    fi
    git -c user.name="local" -c user.email="local@build" cherry-pick "$sha" >/dev/null 2>&1 \
        && echo "  + #$n  $(git log --pretty=%s -1)" \
        || { echo "  ! #$n cherry-pick failed"; exit 1; }
done

echo "[3/3] building llama-server inside $RUNTIME_IMAGE"
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
            -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF >/dev/null && \
        cmake --build build --target llama-server -j$(nproc)'

echo
echo "Done. Binary at $BUILD_DIR/build/bin/llama-server"
echo "Run with: scripts/run-server-aicss.sh"

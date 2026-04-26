#!/usr/bin/env bash
# Run a command inside the intel/vllm xpu container with the project mounted
# at /work and the oneAPI environment sourced. The container has icpx,
# level-zero, and intel-compute-runtime.
#
# Usage:
#   scripts/in-container.sh <command...>
#   scripts/in-container.sh build
#   scripts/in-container.sh test
#   scripts/in-container.sh bench
set -euo pipefail

IMAGE="${VLLM_IMAGE:-docker.io/intel/vllm:0.17.0-xpu}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ "$#" -eq 0 ]; then
    echo "usage: $0 <command...>"
    echo "shortcuts: build | test | bench | shell"
    exit 1
fi

case "${1:-}" in
    build)
        CMD='cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icpx -DXFA_BUILD_PYTHON=OFF && cmake --build build -j'
        ;;
    rebuild)
        CMD='rm -rf build && cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icpx -DXFA_BUILD_PYTHON=OFF && cmake --build build -j'
        ;;
    test)
        CMD='./build/tests/test_flash_attn'
        ;;
    bench)
        CMD='./build/benchmarks/bench_flash_attn'
        ;;
    sdpa-bench)
        CMD='python /work/benchmarks/bench_against_baseline.py --device xpu'
        ;;
    shell)
        CMD='exec bash'
        ;;
    *)
        CMD="$*"
        ;;
esac

# SETVARS_COMPLETED=0 forces re-source even if a previous shell layer set it.
exec podman run --rm -it \
    --device /dev/dri \
    --group-add keep-groups \
    --shm-size=4g \
    -v "$PROJECT_ROOT:/work:z" \
    -w /work \
    -e SETVARS_COMPLETED=0 \
    "$IMAGE" \
    bash -lc ". /opt/intel/oneapi/setvars.sh --force >/dev/null && $CMD"

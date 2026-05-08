#!/usr/bin/env bash
# Wrapper that runs auto-round inside a podman container with XPU
# passthrough, HF cache mount, and (optionally) the live auto-round
# source overlaid via PYTHONPATH. Driven by env vars set by the Nix
# wrapper: AUTOROUND_CONTAINERFILE, AUTOROUND_BUILD_CONTEXT, and
# (when launched from the devshell) AUTOROUND_SOURCE_DIR.
set -euo pipefail

: "${AUTOROUND_CONTAINERFILE:?must be set by nix wrapper}"
: "${AUTOROUND_BUILD_CONTEXT:?must be set by nix wrapper}"

IMAGE_TAG="${AUTOROUND_IMAGE_TAG:-localhost/auto-round-xpu:latest}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
OUTPUT_BASE="${AUTOROUND_OUTPUT_DIR:-$PWD/output}"
SOURCE_DIR="${AUTOROUND_SOURCE_DIR:-}"
USE_SOURCE="${AUTOROUND_USE_SOURCE:-1}"
RECIPE="${AUTOROUND_RECIPE:-light}"

# Auto-detect source-overlay dir when launched outside the devshell
# (e.g. via `nix run`): if the current dir or an ancestor up to 3 levels
# is an auto-round checkout, use it. The devshell hook sets this
# explicitly to $PWD, so this fallback only fires from nix run / direct
# invocation.
if [[ -z "$SOURCE_DIR" ]]; then
    candidate="$PWD"
    for _ in 1 2 3 4; do
        if [[ -f "$candidate/auto_round/__init__.py" && -f "$candidate/setup.cfg" ]]; then
            SOURCE_DIR="$candidate"
            break
        fi
        parent="$(dirname "$candidate")"
        [[ "$parent" == "$candidate" ]] && break
        candidate="$parent"
    done
fi

usage() {
    cat <<EOF
Usage: autoround <model>              # default: quantize with sensible defaults
       autoround quantize <model> [extra auto-round flags]
       autoround shell                # interactive shell inside container
       autoround run -- <cmd...>      # run arbitrary command inside container
       autoround build                # rebuild container image (--no-cache)
       autoround pull                 # pull base image only

Defaults for quantize on 32GB XPU + 96GB RAM (Qwen3.6-35B-A3B class):
  recipe              auto-round-light  (50 iters, lr=5e-3)
  scheme              W4A16             (4-bit sym, group_size=128)
  format              auto_round        (vLLM/INC compatible)
  device              0                 (first XPU)
  batch_size          1
  gradient_accumulate 8
  seqlen              2048
  low_gpu_mem_usage   on
  output              \$AUTOROUND_OUTPUT_DIR/<model-basename>-AutoRound-W4A16

Override any default by appending it: e.g.
  autoround unsloth/Qwen3.6-35B-A3B --scheme W2A16 --seqlen 1024

Environment:
  AUTOROUND_OUTPUT_DIR    output base dir   (default: \$PWD/output)
  AUTOROUND_USE_SOURCE    1 = overlay live source (default), 0 = use pypi
  AUTOROUND_RECIPE        light | default | best
  AUTOROUND_IMAGE_TAG     image tag         (default: localhost/auto-round-xpu:latest)
EOF
}

ensure_image() {
    if ! podman image exists "$IMAGE_TAG"; then
        echo ">>> building $IMAGE_TAG (first run, ~30s on top of cached base)" >&2
        podman build -t "$IMAGE_TAG" -f "$AUTOROUND_CONTAINERFILE" "$AUTOROUND_BUILD_CONTEXT"
    fi
}

build_run_args() {
    mkdir -p "$HF_CACHE" "$OUTPUT_BASE"
    RUN_ARGS=(
        --rm
        --device /dev/dri
        --group-add keep-groups
        --shm-size=16g
        -v "$HF_CACHE:/root/.cache/huggingface:z"
        -v "$OUTPUT_BASE:/output:z"
    )
    if [[ -n "$SOURCE_DIR" && -d "$SOURCE_DIR/auto_round" ]]; then
        RUN_ARGS+=(-v "$SOURCE_DIR:/workspace:z")
        if [[ "$USE_SOURCE" == "1" ]]; then
            RUN_ARGS+=(-e PYTHONPATH=/workspace)
        fi
    fi
    TTY_ARGS=()
    if [[ -t 0 && -t 1 ]]; then
        TTY_ARGS=(-it)
    fi
}

cmd_quantize() {
    if [[ $# -eq 0 ]]; then
        echo "error: missing model name" >&2
        usage >&2
        exit 1
    fi
    local model="$1"; shift

    local recipe_bin
    case "$RECIPE" in
        light)   recipe_bin="auto-round-light" ;;
        default) recipe_bin="auto-round" ;;
        best)    recipe_bin="auto-round-best" ;;
        *) echo "error: AUTOROUND_RECIPE must be light|default|best (got: $RECIPE)" >&2; exit 1 ;;
    esac

    ensure_image
    build_run_args

    echo ">>> $recipe_bin --model $model --scheme W4A16 --format auto_round" >&2
    echo ">>> output base: $OUTPUT_BASE (auto-round will create <model>-w<bits>g<groupsize>/)" >&2

    # auto-round auto-appends <model-name>-w<bits>g<groupsize> to --output_dir,
    # so we point it at the mount root rather than a pre-named subdir.
    exec podman run "${TTY_ARGS[@]}" "${RUN_ARGS[@]}" "$IMAGE_TAG" \
        "$recipe_bin" \
            --model "$model" \
            --scheme W4A16 \
            --format auto_round \
            --device 0 \
            --low_gpu_mem_usage \
            --batch_size 1 \
            --gradient_accumulate_steps 8 \
            --seqlen 2048 \
            --output_dir /output \
            "$@"
}

cmd_shell() {
    ensure_image
    build_run_args
    local workdir=/root
    if [[ -n "$SOURCE_DIR" && -d "$SOURCE_DIR/auto_round" ]]; then
        workdir=/workspace
    fi
    exec podman run "${TTY_ARGS[@]}" "${RUN_ARGS[@]}" --workdir "$workdir" "$IMAGE_TAG" bash
}

cmd_run() {
    ensure_image
    build_run_args
    exec podman run "${TTY_ARGS[@]}" "${RUN_ARGS[@]}" "$IMAGE_TAG" "$@"
}

cmd_build() {
    podman build --no-cache -t "$IMAGE_TAG" -f "$AUTOROUND_CONTAINERFILE" "$AUTOROUND_BUILD_CONTEXT"
}

cmd_pull() {
    local base
    base="$(awk '/^FROM/ {print $2; exit}' "$AUTOROUND_CONTAINERFILE")"
    podman pull "$base"
}

main() {
    local sub="${1:-}"
    case "$sub" in
        quantize) shift; cmd_quantize "$@" ;;
        shell)    shift; cmd_shell ;;
        run)      shift; [[ "${1:-}" == "--" ]] && shift; cmd_run "$@" ;;
        build)    shift; cmd_build ;;
        pull)     shift; cmd_pull ;;
        ""|-h|--help|help) usage ;;
        # Anything else is treated as a model name -> quantize.
        *) cmd_quantize "$@" ;;
    esac
}

main "$@"

#!/usr/bin/env bash
set -euo pipefail

IMAGE="${LLAMA_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-intel}"

echo "Pulling $IMAGE..."
podman pull "$IMAGE"

echo
echo "Image pulled. Inspect with:"
echo "  podman inspect $IMAGE"
echo "  dive $IMAGE"

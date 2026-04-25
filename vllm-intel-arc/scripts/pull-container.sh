#!/usr/bin/env bash
set -euo pipefail

IMAGE="${VLLM_IMAGE:-intel/vllm:0.17.0-xpu}"

echo "Pulling $IMAGE..."
podman pull "$IMAGE"

echo ""
echo "Image pulled. Inspect with:"
echo "  podman inspect $IMAGE"
echo "  dive $IMAGE"

#!/usr/bin/env bash
set -euo pipefail

IMAGE="${VLLM_IMAGE:-intel/vllm@sha256:e961d08135a6a8ef6decd857c6deab7a70eb00e19de21de54cbc0ce05d9a9f43}"

echo "Pulling $IMAGE..."
podman pull "$IMAGE"

echo ""
echo "Image pulled. Inspect with:"
echo "  podman inspect $IMAGE"
echo "  dive $IMAGE"

#!/usr/bin/env bash
# Download the GGUF for a config from Hugging Face into ~/.cache/llamacpp.
#
# Usage:
#   scripts/pull-model.sh [path/to/config.yaml]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${1:-$ROOT/configs/models/qwen3.6-35b-a3b-q4km.yaml}"

if [ ! -f "$CONFIG" ]; then
    echo "config not found: $CONFIG" >&2
    exit 1
fi

# yq is provided by the project nix devshell. Fall back to a minimal grep
# parser when invoked from the bare host shell (no direnv).
if command -v yq >/dev/null 2>&1; then
    REPO=$(yq -r '.repo_id'   "$CONFIG")
    FILE=$(yq -r '.gguf_file' "$CONFIG")
else
    REPO=$(grep -E '^repo_id:'   "$CONFIG" | awk '{print $2}' | tr -d '"')
    FILE=$(grep -E '^gguf_file:' "$CONFIG" | awk '{print $2}' | tr -d '"')
fi

CACHE="$HOME/.cache/llamacpp/models"
mkdir -p "$CACHE"
DEST="$CACHE/$FILE"

if [ -f "$DEST" ]; then
    SIZE=$(stat -c%s "$DEST")
    echo "Already downloaded: $DEST ($((SIZE / 1024 / 1024)) MB)"
    exit 0
fi

URL="https://huggingface.co/${REPO}/resolve/main/${FILE}"
echo "Downloading $URL"
echo "    -> $DEST"

# Use curl -C - for resume; the host shell may not have wget. GGUF files are
# 20+ GB, so resume support matters.
curl -L -C - -o "$DEST.partial" "$URL"
mv "$DEST.partial" "$DEST"

SIZE=$(stat -c%s "$DEST")
echo
echo "Done. $DEST ($((SIZE / 1024 / 1024)) MB)"

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

REPO=$(yq -r '.repo_id'   "$CONFIG")
FILE=$(yq -r '.gguf_file' "$CONFIG")

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

# Use wget with continue support — GGUF files are large (20+ GB).
# huggingface-cli would also work but adds a Python dep we'd have to enter
# the nix shell for; wget keeps this script simple from the host.
wget --continue --progress=bar:force:noscroll -O "$DEST.partial" "$URL"
mv "$DEST.partial" "$DEST"

SIZE=$(stat -c%s "$DEST")
echo
echo "Done. $DEST ($((SIZE / 1024 / 1024)) MB)"

#!/usr/bin/env bash
set -euo pipefail

echo "=== Intel XPU Smoke Test ==="

echo ""
echo "--- Level Zero ---"
if command -v ze_info &>/dev/null; then
  ze_info 2>/dev/null | head -20 || echo "ze_info failed (no GPU driver?)"
else
  echo "ze_info not found, checking for Level Zero library..."
  ls /run/opengl-driver/lib/libze_loader.so* 2>/dev/null || \
  ls /usr/lib/x86_64-linux-gnu/libze_loader.so* 2>/dev/null || \
  echo "Level Zero loader not found in standard paths"
fi

echo ""
echo "--- GPU Detection ---"
if command -v clinfo &>/dev/null; then
  clinfo 2>/dev/null | grep -E "Device Name|Device Type|Driver Version" | head -10 || echo "clinfo failed"
else
  echo "clinfo not available"
fi

echo ""
echo "--- PCIe GPU Devices ---"
lspci 2>/dev/null | grep -i "vga\|display\|3d" || echo "no display devices found via lspci"

echo ""
echo "--- DRM Render Nodes ---"
ls -la /dev/dri/render* 2>/dev/null || echo "no render nodes found"

echo ""
echo "=== Smoke test complete ==="

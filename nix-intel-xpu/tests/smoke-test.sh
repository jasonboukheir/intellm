#!/usr/bin/env bash
set -euo pipefail

echo "=== Intel XPU Smoke Test ==="

echo ""
echo "--- Level Zero ---"
# On NixOS, libraries live in the nix store / opengl-driver paths
ZE_LIB=$(ldconfig -p 2>/dev/null | grep libze_loader || true)
if [ -n "$ZE_LIB" ]; then
  echo "Level Zero loader: $ZE_LIB"
elif [ -f /run/opengl-driver/lib/libze_loader.so ]; then
  echo "Level Zero loader: /run/opengl-driver/lib/libze_loader.so"
else
  # Search nix store and LD_LIBRARY_PATH
  FOUND=$(find /nix/store -maxdepth 3 -name 'libze_loader.so*' 2>/dev/null | head -1 || true)
  if [ -n "$FOUND" ]; then
    echo "Level Zero loader: $FOUND"
  else
    echo "Level Zero loader not found"
  fi
fi

# Enumerate Level Zero devices via a small Python probe
python3 -c "
import ctypes, ctypes.util

lib_path = ctypes.util.find_library('ze_loader')
if not lib_path:
    import glob
    candidates = glob.glob('/nix/store/*/lib/libze_loader.so') + glob.glob('/run/opengl-driver/lib/libze_loader.so')
    lib_path = candidates[0] if candidates else None

if lib_path:
    ze = ctypes.CDLL(lib_path)
    ze.zeInit(0)
    count = ctypes.c_uint32(0)
    ze.zeDriverGet(ctypes.byref(count), None)
    print(f'Level Zero drivers: {count.value}')
    if count.value > 0:
        drivers = (ctypes.c_void_p * count.value)()
        ze.zeDriverGet(ctypes.byref(count), drivers)
        for i in range(count.value):
            dev_count = ctypes.c_uint32(0)
            ze.zeDeviceGet(drivers[i], ctypes.byref(dev_count), None)
            print(f'  Driver {i}: {dev_count.value} device(s)')
else:
    print('Level Zero loader not found for device enumeration')
" 2>/dev/null || echo "Level Zero device enumeration failed"

echo ""
echo "--- GPU Detection (OpenCL) ---"
if command -v clinfo &>/dev/null; then
  clinfo 2>/dev/null | grep -E "Device Name|Device Type|Driver Version|Max compute units|Global memory size" | head -15 || echo "clinfo failed"
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
echo "--- Intel GPU Details ---"
for node in /dev/dri/renderD*; do
  if [ -e "$node" ]; then
    # Use sysfs to identify the device behind each render node
    MINOR=$(stat -c '%T' "$node" 2>/dev/null | xargs printf '%d' 2>/dev/null || true)
    if [ -n "$MINOR" ]; then
      CARD_PATH=$(readlink -f /sys/class/drm/renderD${MINOR}/device 2>/dev/null || true)
      if [ -n "$CARD_PATH" ] && [ -f "$CARD_PATH/vendor" ]; then
        VENDOR=$(cat "$CARD_PATH/vendor" 2>/dev/null)
        DEVICE=$(cat "$CARD_PATH/device" 2>/dev/null)
        if [ "$VENDOR" = "0x8086" ]; then
          MEM=$(cat "$CARD_PATH/mem_info_vram_total" 2>/dev/null || echo "unknown")
          if [ "$MEM" != "unknown" ]; then
            MEM_GB=$(echo "scale=1; $MEM / 1073741824" | bc 2>/dev/null || echo "$MEM bytes")
            MEM_DISPLAY="${MEM_GB} GB"
          else
            MEM_DISPLAY="unknown"
          fi
          echo "$node -> Intel device $DEVICE (VRAM: $MEM_DISPLAY)"
        fi
      fi
    fi
  fi
done 2>/dev/null || echo "(sysfs enumeration not available)"

echo ""
echo "=== Smoke test complete ==="

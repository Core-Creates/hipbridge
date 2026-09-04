#!/usr/bin/env bash
# Install the HIP development headers on a runtime-only ROCm image.
#
# Handles the two things that bit us on AMD Developer Cloud:
#   1. unattended-upgrades holds the dpkg lock right after boot, so the first
#      apt-get install fails with "Could not get lock /var/lib/dpkg/lock-frontend"
#   2. ROCm 7 package names differ from the older hip-dev / rocm-dev names, so
#      the package is searched for rather than guessed.
#
#   bash scripts/install-hip-headers.sh
#
set -eu
say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "What is actually in the ROCm include tree"
for d in /opt/rocm/include /opt/rocm/core-*/include; do
    [ -d "$d" ] || continue
    echo "$d:"
    ls "$d" 2>/dev/null | head -15 | sed 's/^/    /'
    [ -d "$d/hip" ] && echo "    ^ has hip/ already"
done
echo
echo "hip_runtime.h candidates (excluding the openmp_wrappers decoy):"
find /opt -path '*/hip/hip_runtime.h' 2>/dev/null | sed 's/^/  /' || true
echo "  (none listed above means it is genuinely absent)"

say "Waiting for the dpkg lock"
# unattended-upgrades runs on boot and holds this for a minute or two.
for i in $(seq 1 60); do
    if ! fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; then
        echo "lock free"
        break
    fi
    [ "$i" = 1 ] && echo "held by: $(ps -o comm= -p "$(fuser /var/lib/dpkg/lock-frontend 2>/dev/null | tr -d ' ')" 2>/dev/null || echo unattended-upgrades)"
    printf '.'
    sleep 5
done
echo

say "Searching the configured repos for the HIP headers"
apt-get update -qq 2>/dev/null || true
echo "packages providing hip headers:"
apt-cache search --names-only 'hip' 2>/dev/null | grep -iE 'dev|runtime|sdk' | head -20 | sed 's/^/  /' || true
echo
echo "rocm meta/dev packages:"
apt-cache search --names-only 'rocm' 2>/dev/null | grep -iE 'dev|hip' | head -20 | sed 's/^/  /' || true

say "Installing"
INSTALLED=""
for pkg in hip-dev rocm-hip-runtime-dev hip-runtime-amd rocm-dev rocm-hip-sdk hip-base; do
    if apt-cache policy "$pkg" 2>/dev/null | grep -q 'Candidate: [^(]'; then
        echo "trying $pkg"
        if DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$pkg"; then
            INSTALLED="$pkg"
            break
        fi
    fi
done
[ -n "$INSTALLED" ] && echo "installed: $INSTALLED" || echo "no candidate package installed"

say "Result"
HDR="$(find /opt -path '*/hip/hip_runtime.h' 2>/dev/null | head -1 || true)"
if [ -n "$HDR" ]; then
    echo "FOUND: $HDR"
    echo
    echo "Now run:  bash scripts/smoke-hip.sh"
else
    echo "STILL MISSING."
    echo
    echo "The repo list above shows what this image can reach. If nothing there"
    echo "provides hip/hip_runtime.h, relaunching on a ROCm image that bundles"
    echo "PyTorch is faster than assembling a dev toolchain by hand: those ship"
    echo "the headers and torch together."
    exit 1
fi

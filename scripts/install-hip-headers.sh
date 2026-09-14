#!/usr/bin/env bash
# Install HIP development headers that MATCH the installed compiler.
#
# The trap this exists to avoid, observed on an AMD Developer Cloud MI300X:
# `apt-get install libamdhip64-dev` succeeds on Ubuntu 24.04 and installs
# HIP 5.7.1 headers from noble/universe next to a ROCm 7.14 compiler. Nothing
# announces the mismatch. The build fails once with
#
#   amd_warp_functions.h: use of undeclared identifier '__AMDGCN_WAVEFRONT_SIZE'
#
# which is easily worked around by defining the macro, and from then on the skew
# is invisible while every number it produces is quietly suspect.
#
# So this script installs by version rather than by name, and refuses to call
# itself done until the headers it found agree with the compiler that will use
# them. On a ROCm 7 box the package is amdrocm-runtime-dev7.14, not
# libamdhip64-dev and not the hip-dev / rocm-dev names older docs suggest.
#
#   bash scripts/install-hip-headers.sh
#   PURGE_STALE=1 bash scripts/install-hip-headers.sh   # also remove mismatches
#
set -eu
say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# Version of a HIP header tree, as MAJOR.MINOR, or empty if unreadable.
hdr_version() {
    [ -f "$1" ] || return 0
    awk '/#define HIP_VERSION_MAJOR/ { maj=$3 }
         /#define HIP_VERSION_MINOR/ { min=$3 }
         END { if (maj != "") print maj "." min }' "$1"
}

say "The compiler we have to match"
if ! command -v hipcc >/dev/null 2>&1; then
    echo "hipcc is not on PATH. Install the ROCm toolchain before the headers."
    exit 1
fi
hipcc --version 2>/dev/null | head -2 | sed 's/^/  /'
CC_VER="$(hipcc --version 2>/dev/null | sed -n 's/.*HIP version: \([0-9]*\.[0-9]*\).*/\1/p' | head -1)"
[ -n "$CC_VER" ] || CC_VER="unknown"
echo "  compiler HIP version: $CC_VER"

say "Headers present before we start"
for f in $(find /opt /usr/include /usr/local/include -path '*/hip/hip_version.h' 2>/dev/null); do
    V="$(hdr_version "$f")"
    if [ "$V" = "$CC_VER" ]; then
        echo "  $f -> $V  (matches the compiler)"
    else
        echo "  $f -> ${V:-unreadable}  (SKEW: compiler is $CC_VER)"
    fi
done

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

say "Choosing a package for HIP $CC_VER"
apt-get update -qq 2>/dev/null || true

# Version-matched first. AMD's repo carries one package per ROCm minor, so
# asking for the compiler's own version is exact rather than a guess.
CANDIDATES=""
[ "$CC_VER" != "unknown" ] && CANDIDATES="amdrocm-runtime-dev${CC_VER}"
# Then any other amdrocm-runtime-dev, newest first, in case the compiler string
# and the package suffix disagree on some image.
CANDIDATES="$CANDIDATES $(apt-cache search --names-only '^amdrocm-runtime-dev' 2>/dev/null |
    awk '{ print $1 }' | sort -Vr | tr '\n' ' ')"
# Legacy names last. libamdhip64-dev is the Ubuntu-packaged one and on 24.04 it
# is 5.7.1, so it comes after everything else and is verified below like the
# rest rather than trusted.
CANDIDATES="$CANDIDATES hip-dev rocm-hip-runtime-dev rocm-dev libamdhip64-dev"

INSTALLED=""
for pkg in $CANDIDATES; do
    apt-cache policy "$pkg" 2>/dev/null | grep -q 'Candidate: [^(]' || continue
    echo "trying $pkg"
    if DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$pkg" >/dev/null 2>&1; then
        INSTALLED="$pkg"
        echo "  installed: $pkg"
        break
    fi
done
[ -n "$INSTALLED" ] || echo "no candidate package installed; checking what is already here"

# Device bitcode for the specific target. Usually already present under
# /opt/rocm/amdgcn, but the arch metapackage supplies it when it is not.
#
# No fallback arch. This used to end in `|| echo gfx942`, which never fired,
# because `head` succeeds on empty input: the search then ran for a package name
# ending in "-", matched nothing, and skipped the device libraries without a
# word. Had the fallback fired it would have installed MI300X libraries on a
# Radeon. Without an arch, say so and skip.
ARCH_DETECT="${ARCH:-$(rocminfo 2>/dev/null | grep -o 'gfx[0-9a-f]*' | head -1 || true)}"
if [ -z "$ARCH_DETECT" ]; then
    echo "no offload arch detected and ARCH not set; skipping device libraries"
    echo "  (set ARCH=gfxNNNN, as listed by rocminfo, to install them)"
else
    for pkg in $(apt-cache search --names-only "amdrocm-core-dev.*-${ARCH_DETECT}$" 2>/dev/null | awk '{print $1}' | sort -r | head -1); do
        echo "installing device libs for $ARCH_DETECT: $pkg"
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$pkg" >/dev/null 2>&1 || true
    done
fi

say "Does anything now match the compiler"
MATCH=""
STALE=""
for f in $(find /opt /usr/include /usr/local/include -path '*/hip/hip_version.h' 2>/dev/null); do
    V="$(hdr_version "$f")"
    INC="${f%/hip/hip_version.h}"
    if [ "$V" = "$CC_VER" ]; then
        echo "  MATCH  $INC  (HIP $V)"
        [ -z "$MATCH" ] && MATCH="$INC"
    else
        echo "  SKEW   $INC  (HIP ${V:-unreadable}, compiler is $CC_VER)"
        STALE="$STALE $INC"
    fi
done

# A stale tree under /usr/include is worse than merely useless: it is on the
# default include path, so it can win over a correct one that is not.
if [ -n "$STALE" ]; then
    echo
    echo "Mismatched header trees remain:$STALE"
    OWNER="$(dpkg -S /usr/include/hip/hip_version.h 2>/dev/null | cut -d: -f1 || true)"
    if [ -n "${PURGE_STALE:-}" ] && [ -n "$MATCH" ] && [ -n "$OWNER" ]; then
        echo "PURGE_STALE set, removing $OWNER"
        DEBIAN_FRONTEND=noninteractive apt-get remove -y -qq "$OWNER" >/dev/null 2>&1 || true
    elif [ -n "$OWNER" ]; then
        echo "Remove it with:  apt-get remove -y $OWNER"
        echo "(or re-run this script with PURGE_STALE=1, once a match exists)"
    fi
fi

say "Result"
if [ -n "$MATCH" ]; then
    echo "HIP $CC_VER headers at: $MATCH"
    echo
    echo "Now run:  bash scripts/smoke-hip.sh"
    exit 0
fi

echo "NO MATCHING HEADERS for HIP $CC_VER."
echo
echo "What this image can reach:"
apt-cache search --names-only 'amdrocm-runtime' 2>/dev/null | head -10 | sed 's/^/  /' || true
echo
echo "If nothing above provides HIP $CC_VER headers, relaunching on a ROCm image"
echo "that bundles PyTorch is faster than assembling a dev toolchain by hand:"
echo "those ship matching headers, compiler and torch together."
exit 1

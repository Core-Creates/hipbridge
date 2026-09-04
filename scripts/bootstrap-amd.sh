#!/usr/bin/env bash
# Bootstrap hipbridge verification on an AMD ROCm box (MI300X and friends).
#
# Handles a BARE ROCm image: some AMD Developer Cloud images ship hipcc and the
# driver but no pip and no PyTorch. Run scripts/smoke-hip.sh first if you only
# want to know whether the toolchain works; it needs none of this.
#
#   bash scripts/bootstrap-amd.sh
#   LIMIT=24 ARCH=gfx942 bash scripts/bootstrap-amd.sh
#
set -eu

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mFAILED: %s\033[0m\n' "$*"; exit 1; }

say "Environment"
command -v hipcc >/dev/null || die "hipcc not on PATH"
hipcc --version | head -2

# NOTE: `rocminfo | grep -m1` makes grep exit early, SIGPIPEs rocminfo, and under
# `set -o pipefail` that marks the whole pipeline failed even though the value was
# found. Hence no pipefail here, and the grep is unbounded.
if command -v rocminfo >/dev/null 2>&1; then
    DETECTED="$(rocminfo 2>/dev/null | grep -o 'gfx[0-9a-f]*' | head -1 || true)"
else
    DETECTED=""
fi
ARCH="${ARCH:-${DETECTED:-gfx942}}"
echo "arch: $ARCH${DETECTED:+ (detected)}"
command -v rocm-smi >/dev/null 2>&1 && rocm-smi --showproductname 2>/dev/null | head -6 || true

say "Python toolchain"
PY="${PY:-python3}"
command -v "$PY" >/dev/null || die "no python3"
echo "python: $($PY --version 2>&1)"

if ! "$PY" -m pip --version >/dev/null 2>&1; then
    echo "pip missing, installing"
    "$PY" -m ensurepip --upgrade >/dev/null 2>&1 \
        || (command -v apt-get >/dev/null && apt-get update -qq && apt-get install -y -qq python3-pip) \
        || (command -v curl >/dev/null && curl -sS https://bootstrap.pypa.io/get-pip.py | "$PY") \
        || die "could not install pip; try: apt-get install -y python3-pip"
fi
echo "pip: $("$PY" -m pip --version)"

say "PyTorch for ROCm"
if "$PY" -c 'import torch' >/dev/null 2>&1; then
    echo "torch already present, leaving it alone"
else
    ROCM_VER="$(cat /opt/rocm/.info/version 2>/dev/null | cut -d. -f1,2 || true)"
    echo "ROCm version: ${ROCM_VER:-unknown}"
    # Try the matching wheel index first, then walk back. A CUDA or CPU wheel
    # would install cleanly and then silently test the wrong hardware, so the
    # result is checked rather than trusted.
    for IDX in ${ROCM_VER:+rocm$ROCM_VER} rocm6.4 rocm6.3 rocm6.2; do
        echo "trying https://download.pytorch.org/whl/$IDX"
        if "$PY" -m pip install --quiet --index-url "https://download.pytorch.org/whl/$IDX" torch; then
            if "$PY" -c 'import torch,sys; sys.exit(0 if torch.version.hip else 1)' 2>/dev/null; then
                echo "installed ROCm torch from $IDX"
                break
            fi
            echo "  that index gave a non-ROCm build, uninstalling"
            "$PY" -m pip uninstall -y -q torch || true
        fi
    done
    "$PY" -c 'import torch,sys; sys.exit(0 if torch.version.hip else 1)' 2>/dev/null \
        || die "no ROCm PyTorch. Install manually, then re-run:
  $PY -m pip install --index-url https://download.pytorch.org/whl/rocmX.Y torch
Check https://pytorch.org/get-started/locally/ for the index matching ROCm ${ROCM_VER:-?}"
fi

"$PY" - <<'PY'
import torch
print("torch       :", torch.__version__)
print("hip         :", torch.version.hip)
print("devices     :", torch.cuda.device_count())
if torch.cuda.device_count():
    print("device 0    :", torch.cuda.get_device_name(0))
PY

say "Installing hipbridge (torch already satisfied, so untouched)"
"$PY" -m pip install --quiet -e ".[verify,dev]"

say "Capabilities"
hipbridge info

say "Candidate selection (must be the Triton kernel, not the torch fallback)"
"$PY" - <<'PY'
from hipbridge import kernels
from hipbridge.verify import suites
print("kernels usable:", kernels.available())
_, described = suites.candidate_for(suites.ROW_SOFTMAX)
print("candidate     :", described)
PY

say "Verifying against the device (arch=$ARCH)"
hipbridge verify \
    --toolchain hipcc \
    --arch "$ARCH" \
    --limit "${LIMIT:-12}" \
    --require \
    --report amd-verify-report.md

say "Done. Report at amd-verify-report.md"

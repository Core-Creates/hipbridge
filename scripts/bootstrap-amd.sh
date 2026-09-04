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

say "Python environment"
PY_SYS="${PY:-python3}"
command -v "$PY_SYS" >/dev/null || die "no python3"
echo "system python: $($PY_SYS --version 2>&1)"

# Ubuntu 24.04 marks the system Python externally managed (PEP 668), so
# `pip install` into it is refused. A venv is the documented answer and is
# cleaner anyway: it keeps a multi-GB ROCm torch out of the system tree.
VENV="${VENV:-$PWD/.venv}"
if [ ! -x "$VENV/bin/python" ]; then
    echo "creating venv at $VENV"
    if ! "$PY_SYS" -m venv "$VENV" 2>/dev/null; then
        echo "venv module missing, installing python3-venv"
        # unattended-upgrades can hold the dpkg lock for a minute after boot.
        for _ in $(seq 1 60); do
            fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 || break
            printf '.'; sleep 5
        done
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-full             || die "could not install python3-venv"
        "$PY_SYS" -m venv "$VENV" || die "venv creation failed"
    fi
fi
PY="$VENV/bin/python"
echo "venv python: $($PY --version 2>&1)"
"$PY" -m pip install --quiet --upgrade pip

say "PyTorch for ROCm"
if "$PY" -c 'import torch' >/dev/null 2>&1; then
    echo "torch already present in the venv, leaving it alone"
else
    ROCM_VER="$(cat /opt/rocm/.info/version 2>/dev/null | cut -d. -f1,2 || true)"
    if [ -z "$ROCM_VER" ]; then
        # Fall back to the HIP version hipcc reports, e.g. "HIP version: 7.14.x".
        ROCM_VER="$(hipcc --version 2>/dev/null | sed -n 's/.*HIP version: \([0-9]*\.[0-9]*\).*//p' | head -1)"
    fi
    echo "ROCm/HIP version: ${ROCM_VER:-unknown}"

    MAJOR="${ROCM_VER%%.*}"
    if [ "$MAJOR" = "7" ]; then
        INDEXES="rocm$ROCM_VER rocm7.0 rocm6.4 rocm6.3 rocm6.2"
    else
        INDEXES="${ROCM_VER:+rocm$ROCM_VER} rocm6.4 rocm6.3 rocm6.2"
    fi

    # A CUDA or CPU wheel installs perfectly cleanly and then silently tests the
    # wrong hardware, so the result is checked rather than trusted.
    for IDX in $INDEXES; do
        echo "trying https://download.pytorch.org/whl/$IDX"
        if "$PY" -m pip install --quiet --index-url "https://download.pytorch.org/whl/$IDX" torch 2>/dev/null; then
            if "$PY" -c 'import torch,sys; sys.exit(0 if torch.version.hip else 1)' 2>/dev/null; then
                echo "installed ROCm torch from $IDX"
                break
            fi
            echo "  that index gave a non-ROCm build, removing"
            "$PY" -m pip uninstall -y -q torch || true
        fi
    done
    "$PY" -c 'import torch,sys; sys.exit(0 if torch.version.hip else 1)' 2>/dev/null         || die "no ROCm PyTorch. Install into the venv, then re-run:
  $PY -m pip install --index-url https://download.pytorch.org/whl/rocmX.Y torch
See https://pytorch.org/get-started/locally/ for the index matching ROCm ${ROCM_VER:-?}"
fi

"$PY" - <<'PYEOF'
import torch
print("torch       :", torch.__version__)
print("hip         :", torch.version.hip)
print("devices     :", torch.cuda.device_count())
if torch.cuda.device_count():
    print("device 0    :", torch.cuda.get_device_name(0))
PYEOF

say "Installing hipbridge into the venv"
"$PY" -m pip install --quiet -e ".[verify,dev]"

say "Capabilities"
"$VENV/bin/hipbridge" info

say "Candidate selection (must be the Triton kernel, not the torch fallback)"
"$PY" - <<'PY'
from hipbridge import kernels
from hipbridge.verify import suites
print("kernels usable:", kernels.available())
_, described = suites.candidate_for(suites.ROW_SOFTMAX)
print("candidate     :", described)
PY

say "Verifying against the device (arch=$ARCH)"
"$VENV/bin/hipbridge" verify \
    --toolchain hipcc \
    --arch "$ARCH" \
    --limit "${LIMIT:-12}" \
    --require \
    --report amd-verify-report.md

say "Done. Report at amd-verify-report.md"

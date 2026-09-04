#!/usr/bin/env bash
# Bootstrap hipbridge verification on an AMD ROCm box (MI300X and friends).
#
# Paste this into a fresh AMD Developer Cloud instance. It is deliberately
# conservative about PyTorch: ROCm images ship a ROCm-built torch plus
# pytorch-triton-rocm, and reinstalling either from PyPI would replace them with
# NVIDIA builds and break the GPU you are paying for. Nothing here upgrades torch.
#
#   bash scripts/bootstrap-amd.sh
#
set -euo pipefail

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "Environment"
command -v hipcc     >/dev/null && hipcc --version | head -2 || { echo "hipcc MISSING"; exit 1; }
command -v rocminfo  >/dev/null && rocminfo | grep -m1 -o 'gfx[0-9a-f]*' || echo "rocminfo missing"
command -v rocm-smi  >/dev/null && rocm-smi --showproductname 2>/dev/null | head -5 || true

say "Existing PyTorch (must NOT be replaced)"
python3 - <<'PY'
try:
    import torch
    print("torch        :", torch.__version__)
    print("hip version  :", getattr(torch.version, "hip", None))
    print("device count :", torch.cuda.device_count())
    print("device 0     :", torch.cuda.get_device_name(0) if torch.cuda.device_count() else "none")
except Exception as e:
    print("torch not importable:", e)
PY

say "Installing hipbridge (core + verify; torch already satisfied, so untouched)"
# --no-build-isolation is not needed; setuptools is standard. We rely on pip
# leaving an already-satisfied torch alone rather than passing --no-deps, so
# libclang still installs.
python3 -m pip install --quiet -e ".[verify,dev]"

say "Capabilities"
hipbridge info

say "Sanity: the suite must select the Triton kernel, not the torch fallback"
python3 - <<'PY'
from hipbridge import kernels
from hipbridge.verify import suites
print("kernels usable:", kernels.available())
_, described = suites.candidate_for(suites.ROW_SOFTMAX)
print("candidate     :", described)
PY

ARCH="${ARCH:-$(rocminfo 2>/dev/null | grep -m1 -o 'gfx[0-9a-f]*' || echo gfx942)}"
say "Verifying against the device (arch=$ARCH)"
hipbridge verify \
    --toolchain hipcc \
    --arch "$ARCH" \
    --limit "${LIMIT:-12}" \
    --require \
    --report amd-verify-report.md

say "Done. Report at amd-verify-report.md"

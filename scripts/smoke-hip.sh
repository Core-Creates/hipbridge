#!/usr/bin/env bash
# Zero-dependency check that hipcc can build and run what hipbridge generates.
#
# No Python, no pip, no PyTorch. Compiles the same driver shape that
# verify/reference.py emits, runs row_softmax on device, and self-checks the
# result against a CPU computation. Use this first on a fresh ROCm box: it
# isolates "does the toolchain work" from "is the Python environment set up".
#
#   bash scripts/smoke-hip.sh [gfx942]
#
set -eu

ARCH="${1:-${ARCH:-}}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

command -v hipcc >/dev/null || { echo "FAIL: hipcc not on PATH"; exit 1; }
echo "hipcc: $(hipcc --version | head -1)"

if [ -z "$ARCH" ] && command -v rocminfo >/dev/null 2>&1; then
    ARCH="$(rocminfo 2>/dev/null | grep -o 'gfx[0-9a-f]*' | head -1 || true)"
fi
[ -n "$ARCH" ] && echo "arch: $ARCH" || echo "arch: (letting hipcc choose)"

cat > "$WORK/smoke.hip.cpp" <<'HIP'
#include <cstdio>
#include <cmath>
#include <vector>
#include <hip/hip_runtime.h>

// Verbatim from examples/row_softmax.cu. It compiles under hipcc unchanged
// because it uses only constructs HIP spells identically.
__global__ void row_softmax(const float *in, float *out, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float *ri = in + row * cols;
    float *ro = out + row * cols;
    float m = -1e20f;
    for (int i = 0; i < cols; i++) m = fmaxf(m, ri[i]);
    float s = 0.0f;
    for (int i = 0; i < cols; i++) { float v = expf(ri[i] - m); ro[i] = v; s += v; }
    for (int i = 0; i < cols; i++) ro[i] /= s;
}

int main() {
    const int rows = 64, cols = 128, n = rows * cols;
    std::vector<float> h_in(n), h_out(n);
    for (int i = 0; i < n; i++) h_in[i] = sinf(i * 0.37f) * 3.0f;

    float *d_in = nullptr, *d_out = nullptr;
    if (hipMalloc(&d_in, n * sizeof(float)) != hipSuccess) { printf("FAIL: hipMalloc\n"); return 2; }
    if (hipMalloc(&d_out, n * sizeof(float)) != hipSuccess) { printf("FAIL: hipMalloc\n"); return 2; }
    (void)hipMemcpy(d_in, h_in.data(), n * sizeof(float), hipMemcpyHostToDevice);

    // Sentinel, exactly as the generated driver does: catches a kernel that
    // never stores, which would otherwise pass on whatever the allocator gave.
    std::vector<float> sentinel(n, -12345.0f);
    (void)hipMemcpy(d_out, sentinel.data(), n * sizeof(float), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(row_softmax, dim3(rows,1,1), dim3(1,1,1), 0, 0, d_in, d_out, rows, cols);
    (void)hipDeviceSynchronize();
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) { printf("FAIL: kernel error: %s\n", hipGetErrorString(err)); return 3; }

    (void)hipMemcpy(h_out.data(), d_out, n * sizeof(float), hipMemcpyDeviceToHost);

    int untouched = 0;
    for (int i = 0; i < n; i++) if (h_out[i] == -12345.0f) untouched++;
    if (untouched) { printf("FAIL: %d elements never written\n", untouched); return 4; }

    double worst_sum = 0.0, worst_val = 0.0;
    for (int r = 0; r < rows; r++) {
        const float *ri = h_in.data() + r * cols;
        double m = -1e30, s = 0.0, rowsum = 0.0;
        for (int i = 0; i < cols; i++) m = fmax(m, (double)ri[i]);
        for (int i = 0; i < cols; i++) s += exp((double)ri[i] - m);
        for (int i = 0; i < cols; i++) {
            double want = exp((double)ri[i] - m) / s;
            worst_val = fmax(worst_val, fabs(want - (double)h_out[r*cols+i]));
            rowsum += h_out[r*cols+i];
        }
        worst_sum = fmax(worst_sum, fabs(rowsum - 1.0));
    }
    printf("max |row sum - 1| : %.3e\n", worst_sum);
    printf("max abs error     : %.3e  (vs float64 CPU)\n", worst_val);
    if (worst_val > 1e-5 || worst_sum > 1e-4) { printf("FAIL: numerically wrong\n"); return 5; }

    printf("\nPASS: hipcc built it, MI300X ran it, the numbers are right.\n");
    return 0;
}
HIP

FLAGS="-O2"
[ -n "$ARCH" ] && FLAGS="$FLAGS --offload-arch=$ARCH"

# Some ROCm images use a component-split layout (/opt/rocm/core-<ver>/...) where
# hipcc does not find its own headers, giving:
#   fatal error: 'hip/hip_runtime.h' file not found
# Locate them and put them on the include path rather than guessing.
echo "locating hip/hip_runtime.h"
HIP_INC=""

# Ask dpkg first. If the package is installed it knows exactly where the header
# went, which beats guessing prefixes. libamdhip64-dev is Debian-packaged, so it
# installs under /usr/include, NOT under /opt/rocm as ROCm tarball installs do.
#
# Order matters. On Ubuntu 24.04 libamdhip64-dev is HIP 5.7.1, which installs
# happily beside a ROCm 7 compiler, so the AMD-packaged amdrocm-runtime-dev is
# asked first and the Ubuntu one is the fallback. Whatever wins gets its version
# checked against the compiler below.
if command -v dpkg >/dev/null 2>&1; then
    for pkg in $(dpkg-query -W -f='${Package}\n' 'amdrocm-runtime-dev*' 2>/dev/null | sort -Vr) \
               libamdhip64-dev hip-dev rocm-hip-runtime-dev; do
        F="$(dpkg -L "$pkg" 2>/dev/null | grep -m1 '/hip/hip_runtime\.h$' || true)"
        if [ -n "$F" ]; then
            HIP_INC="${F%/hip/hip_runtime.h}"
            echo "  dpkg says $pkg provides it"
            break
        fi
    done
fi

if [ -z "$HIP_INC" ]; then
    for d in "${ROCM_PATH:-}/include" /opt/rocm/include /opt/rocm-*/include              /opt/rocm/*/include /usr/include /usr/local/include; do
        if [ -f "$d/hip/hip_runtime.h" ]; then HIP_INC="$d"; break; fi
    done
fi
if [ -z "$HIP_INC" ]; then
    FOUND="$(find /opt /usr/include /usr/local/include -path '*/hip/hip_runtime.h' 2>/dev/null | head -1 || true)"
    [ -n "$FOUND" ] && HIP_INC="${FOUND%/hip/hip_runtime.h}"
fi

if [ -n "$HIP_INC" ]; then
    echo "  found: $HIP_INC/hip/hip_runtime.h"
    # Say out loud whether these headers belong to this compiler. A skew still
    # builds once the wavefront macro is defined by hand, so without this check
    # it stays invisible and every number downstream inherits it.
    CC_VER="$(hipcc --version 2>/dev/null | sed -n 's/.*HIP version: \([0-9]*\.[0-9]*\).*/\1/p' | head -1)"
    HDR_VER="$(awk '/#define HIP_VERSION_MAJOR/ { maj=$3 }
                    /#define HIP_VERSION_MINOR/ { min=$3 }
                    END { if (maj != "") print maj "." min }' \
               "$HIP_INC/hip/hip_version.h" 2>/dev/null)"
    if [ -n "$CC_VER" ] && [ -n "$HDR_VER" ] && [ "$CC_VER" != "$HDR_VER" ]; then
        echo "  WARNING: headers are HIP $HDR_VER but the compiler is HIP $CC_VER."
        echo "           Numbers from a mismatched build are not worth publishing."
        echo "           Fix with: bash scripts/install-hip-headers.sh"
    else
        echo "  version: HIP ${HDR_VER:-unknown}, matching the compiler"
    fi
    FLAGS="$FLAGS -I$HIP_INC"
    # Deliberately NOT deriving --rocm-path from the header location. With
    # libamdhip64-dev the headers are under /usr while the toolchain is under
    # /opt/rocm, so passing --rocm-path=/usr sends hipcc looking for
    # /usr/lib/llvm/bin/clang++ and it stops finding its own working compiler.
    # hipcc resolves its toolchain correctly on its own; only the headers were
    # ever missing.
else
    echo "  NOT FOUND"
    echo
    echo "The HIP development headers are not installed. hipcc and the runtime"
    echo "are present, but the headers ship as a separate package on some images."
    echo
    echo "Note: a file called hip_runtime.h under .../openmp_wrappers/ is NOT the"
    echo "one. That is clang's OpenMP offload shim and it shares the filename."
    echo "The real header is at <prefix>/include/hip/hip_runtime.h."
    echo
    echo "ROCm components present:"
    ls -d /opt/rocm*/ /opt/rocm/*/ 2>/dev/null | sed 's/^/  /' || echo "  none"
    echo
    if command -v apt-get >/dev/null 2>&1; then
        echo "Candidate packages (apt), version-matched first:"
        CC_VER="$(hipcc --version 2>/dev/null | sed -n 's/.*HIP version: \([0-9]*\.[0-9]*\).*/\1/p' | head -1)"
        for pkg in ${CC_VER:+amdrocm-runtime-dev$CC_VER} hip-dev rocm-hip-runtime-dev rocm-dev hip-runtime-amd; do
            if apt-cache show "$pkg" >/dev/null 2>&1; then
                echo "    AVAILABLE  apt-get install -y $pkg"
            else
                echo "    not in configured repos: $pkg"
            fi
        done
        echo
        echo "If none are available, the ROCm apt repository is not configured."
    fi
    echo
    echo "Often faster: relaunch the instance on a ROCm image that includes"
    echo "PyTorch. Those ship the HIP headers and torch together, which also"
    echo "removes the separate pip/PyTorch setup this box needs."
    exit 6
fi

# Wavefront width for the target. CDNA (gfx9xx) is 64 wide; RDNA (gfx10xx and
# later) is 32. Used only if the header/compiler mismatch below needs papering.
case "$ARCH" in
    gfx9*)  WAVE=64 ;;
    gfx1*)  WAVE=32 ;;
    *)      WAVE=64 ;;
esac

echo "compiling: hipcc $FLAGS"
if hipcc $FLAGS "$WORK/smoke.hip.cpp" -o "$WORK/smoke" 2>"$WORK/err.txt"; then
    :
elif grep -q '__AMDGCN_WAVEFRONT_SIZE' "$WORK/err.txt"; then
    # The apt headers (/usr/include/hip) and the compiler (/opt/rocm/core-*) come
    # from different ROCm versions. The headers reference __AMDGCN_WAVEFRONT_SIZE,
    # which this clang does not predefine under that spelling. Supplying it for
    # the detected arch is correct and lets the build proceed; the proper fix is
    # to install headers matching the compiler version.
    echo
    echo "  header/compiler version mismatch: __AMDGCN_WAVEFRONT_SIZE undefined"
    echo "  retrying with -D__AMDGCN_WAVEFRONT_SIZE=$WAVE (correct for $ARCH)"
    echo "  this papers over the skew; fix it with scripts/install-hip-headers.sh"
    FLAGS="$FLAGS -D__AMDGCN_WAVEFRONT_SIZE=$WAVE"
    if ! hipcc $FLAGS "$WORK/smoke.hip.cpp" -o "$WORK/smoke" 2>"$WORK/err2.txt"; then
        echo
        echo "still failing:"
        tail -25 "$WORK/err2.txt"
        echo
        echo "The headers in $HIP_INC do not match the compiler in"
        echo "$(dirname "$(dirname "$(command -v hipcc)")")."
        echo "Install headers matching the compiler:"
        echo "    bash scripts/install-hip-headers.sh"
        echo "which picks amdrocm-runtime-dev<version> for this compiler. Do NOT"
        echo "reach for libamdhip64-dev on Ubuntu 24.04: it is HIP 5.7.1 and it"
        echo "is what put the wrong headers here in the first place."
        exit 7
    fi
else
    echo "compile failed:"
    tail -25 "$WORK/err.txt"
    exit 7
fi
echo "running..."
"$WORK/smoke"

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
    hipMemcpy(d_in, h_in.data(), n * sizeof(float), hipMemcpyHostToDevice);

    // Sentinel, exactly as the generated driver does: catches a kernel that
    // never stores, which would otherwise pass on whatever the allocator gave.
    std::vector<float> sentinel(n, -12345.0f);
    hipMemcpy(d_out, sentinel.data(), n * sizeof(float), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(row_softmax, dim3(rows,1,1), dim3(1,1,1), 0, 0, d_in, d_out, rows, cols);
    hipDeviceSynchronize();
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) { printf("FAIL: kernel error: %s\n", hipGetErrorString(err)); return 3; }

    hipMemcpy(h_out.data(), d_out, n * sizeof(float), hipMemcpyDeviceToHost);

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
echo "compiling: hipcc $FLAGS"
hipcc $FLAGS "$WORK/smoke.hip.cpp" -o "$WORK/smoke"
echo "running..."
"$WORK/smoke"

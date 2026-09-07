// RMSNorm as it is usually first written: one block per row, one thread, one
// serial accumulation of squares.
//
// Cheaper than LayerNorm because it skips the mean entirely, and less accurate
// per element for the same reason it is cheaper: everything rides on a single
// serial sum of x*x, where large values dominate and small ones fall off the
// end of the accumulator.
//
// No gamma: the affine step is deliberately absent so the whole kernel is
// covered by a single-input proof.

// Element type comes from the driver, which defines HB_SCALAR for the run.
// Loads and stores use it; the arithmetic in between is float, because a half
// precision sum of a long row loses most of its mantissa and the point of this
// kernel is to be a fair reference, not a fast one.
#ifndef HB_SCALAR
#define HB_SCALAR float
#endif
__global__ void rms_norm(const HB_SCALAR *in, HB_SCALAR *out, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const HB_SCALAR *ri = in + (size_t)row * cols;
    HB_SCALAR *ro = out + (size_t)row * cols;

    float acc = 0.0f;
    for (int i = 0; i < cols; i++)
        acc += (float)ri[i] * (float)ri[i];

    float inv = rsqrtf(acc / (float)cols + 1e-5f);

    for (int i = 0; i < cols; i++)
        ro[i] = (HB_SCALAR)((float)ri[i] * inv);
}

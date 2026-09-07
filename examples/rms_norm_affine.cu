// RMSNorm with a learned scale, the form most current LLMs use. No mean, no
// shift: one accumulation of squares, one reciprocal square root, one weight.
//
// Argument order follows the generated driver: every input buffer, then the
// output, then the scalars. So (in, gamma, out, rows, cols).

// Element type comes from the driver, which defines HB_SCALAR for the run.
// Loads and stores use it; the arithmetic in between is float, because a half
// precision sum of a long row loses most of its mantissa and the point of this
// kernel is to be a fair reference, not a fast one.
#ifndef HB_SCALAR
#define HB_SCALAR float
#endif
__global__ void rms_norm_affine(const HB_SCALAR *in, const HB_SCALAR *gamma,
                                HB_SCALAR *out, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const HB_SCALAR *ri = in + (size_t)row * cols;
    HB_SCALAR *ro = out + (size_t)row * cols;

    float acc = 0.0f;
    for (int i = 0; i < cols; i++)
        acc += (float)ri[i] * (float)ri[i];

    float inv = rsqrtf(acc / (float)cols + 1e-5f);

    for (int i = 0; i < cols; i++)
        ro[i] = (HB_SCALAR)((float)ri[i] * inv * (float)gamma[i]);
}

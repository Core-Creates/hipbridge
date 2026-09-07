// LayerNorm as a transformer actually uses it: normalise, then scale by a
// learned gamma and shift by a learned beta.
//
// This is the kernel the harness could not verify until it could carry more
// than one input tensor. Proving the normalisation alone and then shipping a
// substitution that also multiplies by weights would be proving one program and
// deploying another.
//
// Argument order matters and is not free: the generated driver passes every
// input buffer, then the output, then the scalars. So (in, gamma, beta, out,
// rows, cols) and nothing else.

// Element type comes from the driver, which defines HB_SCALAR for the run.
// Loads and stores use it; the arithmetic in between is float, because a half
// precision sum of a long row loses most of its mantissa and the point of this
// kernel is to be a fair reference, not a fast one.
#ifndef HB_SCALAR
#define HB_SCALAR float
#endif
__global__ void layer_norm_affine(const HB_SCALAR *in, const HB_SCALAR *gamma, const HB_SCALAR *beta,
                                  HB_SCALAR *out, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const HB_SCALAR *ri = in + (size_t)row * cols;
    HB_SCALAR *ro = out + (size_t)row * cols;

    float sum = 0.0f;
    for (int i = 0; i < cols; i++)
        sum += (float)ri[i];
    float mean = sum / (float)cols;

    float acc = 0.0f;
    for (int i = 0; i < cols; i++) {
        float d = (float)ri[i] - mean;
        acc += d * d;
    }
    float inv = rsqrtf(acc / (float)cols + 1e-5f);

    for (int i = 0; i < cols; i++)
        ro[i] = (HB_SCALAR)(((float)ri[i] - mean) * inv * (float)gamma[i] + (float)beta[i]);
}

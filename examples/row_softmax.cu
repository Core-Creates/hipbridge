// One block per row, three serial passes. Must NOT be classified elementwise:
// dropping the max-subtraction pass is silent on small inputs and catastrophic
// on large logits.

// Element type comes from the driver, which defines HB_SCALAR for the run.
// Loads and stores use it; the arithmetic in between is float, because a half
// precision sum of a long row loses most of its mantissa and the point of this
// kernel is to be a fair reference, not a fast one.
#ifndef HB_SCALAR
#define HB_SCALAR float
#endif
__global__ void row_softmax(const HB_SCALAR *in, HB_SCALAR *out, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const HB_SCALAR *ri = in + row * cols;
    HB_SCALAR *ro = out + row * cols;

    float m = -1e20f;
    for (int i = 0; i < cols; i++)
        m = fmaxf(m, (float)ri[i]);

    float s = 0.0f;
    for (int i = 0; i < cols; i++) {
        float v = expf((float)ri[i] - m);
        ro[i] = (HB_SCALAR)v;
        s += v;
    }

    for (int i = 0; i < cols; i++)
        ro[i] = (HB_SCALAR)((float)ro[i] / s);
}

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
__global__ void rms_norm(const float *in, float *out, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float *ri = in + (size_t)row * cols;
    float *ro = out + (size_t)row * cols;

    float acc = 0.0f;
    for (int i = 0; i < cols; i++)
        acc += ri[i] * ri[i];

    float inv = rsqrtf(acc / (float)cols + 1e-5f);

    for (int i = 0; i < cols; i++)
        ro[i] = ri[i] * inv;
}

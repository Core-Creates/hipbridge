// One block per row, one thread per block, everything serial. The shape a
// CUDA kernel takes when it was written to be obvious rather than fast.
//
// Two serial accumulations over the row, which is where its accuracy goes: a
// serial sum grows rounding error as O(n) where a pairwise one grows it as
// O(log n), and the variance pass sums squares, which widens the range being
// accumulated and loses more of it.
//
// No gamma or beta: the affine step is deliberately absent so the whole kernel
// is covered by a single-input proof.
__global__ void layer_norm(const float *in, float *out, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float *ri = in + (size_t)row * cols;
    float *ro = out + (size_t)row * cols;

    float sum = 0.0f;
    for (int i = 0; i < cols; i++)
        sum += ri[i];
    float mean = sum / (float)cols;

    float acc = 0.0f;
    for (int i = 0; i < cols; i++) {
        float d = ri[i] - mean;
        acc += d * d;
    }
    float inv = rsqrtf(acc / (float)cols + 1e-5f);

    for (int i = 0; i < cols; i++)
        ro[i] = (ri[i] - mean) * inv;
}

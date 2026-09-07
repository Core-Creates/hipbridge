// Affine LayerNorm with the row reduced in parallel, so the benchmark has a
// baseline that can win. Same maths as layer_norm_affine.cu: one block per row,
// 256 threads, two shared-memory tree reductions, then scale and shift.

// Element type comes from the driver, which defines HB_SCALAR for the run.
// Loads and stores use it; the arithmetic in between is float, because a half
// precision sum of a long row loses most of its mantissa and the point of this
// kernel is to be a fair reference, not a fast one.
#ifndef HB_SCALAR
#define HB_SCALAR float
#endif
#define HB_TILE 256

__global__ void layer_norm_affine_tuned(const HB_SCALAR *in, const HB_SCALAR *gamma, const HB_SCALAR *beta,
                                        HB_SCALAR *out, int rows, int cols) {
    __shared__ float red[HB_TILE];

    int row = blockIdx.x;
    if (row >= rows) return;

    const HB_SCALAR *ri = in + (size_t)row * cols;
    HB_SCALAR *ro = out + (size_t)row * cols;
    int t = threadIdx.x;

    float acc = 0.0f;
    for (int i = t; i < cols; i += HB_TILE)
        acc += (float)ri[i];
    red[t] = acc;
    __syncthreads();
    for (int s = HB_TILE / 2; s > 0; s >>= 1) {
        if (t < s) red[t] += red[t + s];
        __syncthreads();
    }
    float mean = red[0] / (float)cols;
    __syncthreads();

    acc = 0.0f;
    for (int i = t; i < cols; i += HB_TILE) {
        float d = (float)ri[i] - mean;
        acc += d * d;
    }
    red[t] = acc;
    __syncthreads();
    for (int s = HB_TILE / 2; s > 0; s >>= 1) {
        if (t < s) red[t] += red[t + s];
        __syncthreads();
    }
    float inv = rsqrtf(red[0] / (float)cols + 1e-5f);
    __syncthreads();

    for (int i = t; i < cols; i += HB_TILE)
        ro[i] = (HB_SCALAR)(((float)ri[i] - mean) * inv * (float)gamma[i] + (float)beta[i]);
}

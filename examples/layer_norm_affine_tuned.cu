// Affine LayerNorm with the row reduced in parallel, so the benchmark has a
// baseline that can win. Same maths as layer_norm_affine.cu: one block per row,
// 256 threads, two shared-memory tree reductions, then scale and shift.
#define HB_TILE 256

__global__ void layer_norm_affine_tuned(const float *in, const float *gamma, const float *beta,
                                        float *out, int rows, int cols) {
    __shared__ float red[HB_TILE];

    int row = blockIdx.x;
    if (row >= rows) return;

    const float *ri = in + (size_t)row * cols;
    float *ro = out + (size_t)row * cols;
    int t = threadIdx.x;

    float acc = 0.0f;
    for (int i = t; i < cols; i += HB_TILE)
        acc += ri[i];
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
        float d = ri[i] - mean;
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
        ro[i] = (ri[i] - mean) * inv * gamma[i] + beta[i];
}

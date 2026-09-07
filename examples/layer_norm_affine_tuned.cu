// Affine LayerNorm written the way someone who knows the hardware would write it, so
// the benchmark has a baseline that can actually win. Same maths as
// layer_norm.cu, parallel within the row instead of serial.
//
// One pass over the row rather than two, using Welford's algorithm. The obvious
// formulation reads the row to find the mean, reads it again to accumulate
// squared deviations, and reads it a third time to normalise. Welford carries
// count, mean and the sum of squared deviations together, and its parallel
// combination is associative, so a block reduces partial (n, mean, M2) triples
// the same way it would reduce a plain sum:
//
//     n = na + nb
//     mean = mean_a + (mean_b - mean_a) * nb / n
//     M2 = M2a + M2b + (mean_b - mean_a)^2 * na * nb / n
//
// Two reads instead of three, on a kernel that is memory bound, and it is more
// accurate than accumulating sum and sum-of-squares separately, which loses
// precision to cancellation when the mean is large relative to the spread.
//
// One block per row, 256 threads, strided loads. Only constructs HIP and CUDA
// spell identically, so hipcc and nvcc both take it unchanged.

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
    __shared__ float red_n[HB_TILE];
    __shared__ float red_mean[HB_TILE];
    __shared__ float red_m2[HB_TILE];

    int row = blockIdx.x;
    if (row >= rows) return;

    const HB_SCALAR *ri = in + (size_t)row * cols;
    HB_SCALAR *ro = out + (size_t)row * cols;
    int t = threadIdx.x;

    float n = 0.0f, mean = 0.0f, m2 = 0.0f;
    for (int i = t; i < cols; i += HB_TILE) {
        float v = (float)ri[i];
        n += 1.0f;
        float d = v - mean;
        mean += d / n;
        m2 += d * (v - mean);
    }
    red_n[t] = n;
    red_mean[t] = mean;
    red_m2[t] = m2;
    __syncthreads();

    for (int stride = HB_TILE / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            float na = red_n[t], nb = red_n[t + stride];
            float total = na + nb;
            if (total > 0.0f) {
                float delta = red_mean[t + stride] - red_mean[t];
                red_mean[t] = red_mean[t] + delta * (nb / total);
                red_m2[t] = red_m2[t] + red_m2[t + stride] + delta * delta * (na * nb / total);
                red_n[t] = total;
            }
        }
        __syncthreads();
    }
    float row_mean = red_mean[0];
    float inv = rsqrtf(red_m2[0] / (float)cols + 1e-5f);
    __syncthreads();

    for (int i = t; i < cols; i += HB_TILE)
        ro[i] = (HB_SCALAR)(((float)ri[i] - row_mean) * inv * (float)gamma[i] + (float)beta[i]);
}

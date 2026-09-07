// The same softmax, written the way someone who knows the hardware would write
// it. Exists so the benchmark has an honest baseline: beating row_softmax.cu is
// easy, because that kernel launches one thread per block and leaves the device
// idle. If a competent HIP kernel closes the gap, the substitution is not worth
// making and this file is how we find that out.
//
// Online softmax (Milakov and Gimelshein), so the row is read twice rather than
// three times. The obvious formulation makes three passes: find the maximum,
// exponentiate and sum, then divide. This one carries a running maximum and a
// running sum together, rescaling the sum whenever the maximum moves:
//
//     m' = max(m, x)      s' = s * exp(m - m') + exp(x - m')
//
// which is associative, so a block can reduce partial (m, s) pairs the same way
// it would reduce a plain sum. Softmax at these sizes is memory bound, so a
// third of the traffic is close to a third of the time.
//
// The first version of this file did make three passes, and torch beat it by
// 2.3x, which made "2.9x against a competent kernel" softer than it sounded.
// A baseline that is merely better than terrible is not a baseline.
//
// One block per row, 256 threads, strided loads, shared-memory tree reductions.
// Deliberately portable: only constructs HIP and CUDA spell identically, so the
// same source compiles under hipcc and nvcc.

// Element type comes from the driver, which defines HB_SCALAR for the run.
// Loads and stores use it; the arithmetic in between is float, because a half
// precision sum of a long row loses most of its mantissa and the point of this
// kernel is to be a fair reference, not a fast one.
#ifndef HB_SCALAR
#define HB_SCALAR float
#endif

#define HB_TILE 256

__global__ void row_softmax_tuned(const HB_SCALAR *in, HB_SCALAR *out, int rows, int cols) {
    __shared__ float red_m[HB_TILE];
    __shared__ float red_s[HB_TILE];

    int row = blockIdx.x;
    if (row >= rows) return;

    const HB_SCALAR *ri = in + (size_t)row * cols;
    HB_SCALAR *ro = out + (size_t)row * cols;
    int t = threadIdx.x;

    // Pass 1: one walk of the row carrying both the maximum and the sum.
    float m = -1e20f;
    float s = 0.0f;
    for (int i = t; i < cols; i += HB_TILE) {
        float v = (float)ri[i];
        float mn = fmaxf(m, v);
        s = s * expf(m - mn) + expf(v - mn);
        m = mn;
    }
    red_m[t] = m;
    red_s[t] = s;
    __syncthreads();

    // Combine partials pairwise. Rescaling the smaller sum onto the larger
    // maximum is what makes this reduction associative, and it never
    // exponentiates a positive number, so it cannot overflow.
    for (int stride = HB_TILE / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            float m_other = red_m[t + stride];
            float s_other = red_s[t + stride];
            float m_new = fmaxf(red_m[t], m_other);
            red_s[t] = red_s[t] * expf(red_m[t] - m_new) + s_other * expf(m_other - m_new);
            red_m[t] = m_new;
        }
        __syncthreads();
    }
    float total_max = red_m[0];
    float total_sum = red_s[0];
    __syncthreads();

    // Pass 2: exponentiate and normalise in one go, so nothing is written twice.
    float inv = 1.0f / total_sum;
    for (int i = t; i < cols; i += HB_TILE)
        ro[i] = (HB_SCALAR)(expf((float)ri[i] - total_max) * inv);
}

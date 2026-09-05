// The same softmax, written the way someone who knows the hardware would write
// it. Exists so the benchmark has an honest baseline: beating row_softmax.cu is
// easy, because that kernel launches one thread per block and leaves the device
// idle. If a competent HIP kernel closes the gap, the substitution is not worth
// making and this file is how we find that out.
//
// One block per row, 256 threads, strided loads, two shared-memory tree
// reductions. Same three passes and same maths as the naive version, parallel
// within the row instead of serial. Deliberately portable: only constructs HIP
// and CUDA spell identically, so the same source compiles under hipcc and nvcc.
#define HB_TILE 256

__global__ void row_softmax_tuned(const float *in, float *out, int rows, int cols) {
    __shared__ float red[HB_TILE];

    int row = blockIdx.x;
    if (row >= rows) return;

    const float *ri = in + (size_t)row * cols;
    float *ro = out + (size_t)row * cols;
    int t = threadIdx.x;

    // Pass 1: row maximum, strided scan then a tree reduction.
    float m = -1e20f;
    for (int i = t; i < cols; i += HB_TILE)
        m = fmaxf(m, ri[i]);
    red[t] = m;
    __syncthreads();
    for (int s = HB_TILE / 2; s > 0; s >>= 1) {
        if (t < s) red[t] = fmaxf(red[t], red[t + s]);
        __syncthreads();
    }
    m = red[0];
    __syncthreads();

    // Pass 2: exponentials and their sum. Pairwise here, which is also why this
    // kernel disagrees with the serial original in the last few ULPs.
    float acc = 0.0f;
    for (int i = t; i < cols; i += HB_TILE) {
        float v = expf(ri[i] - m);
        ro[i] = v;
        acc += v;
    }
    red[t] = acc;
    __syncthreads();
    for (int s = HB_TILE / 2; s > 0; s >>= 1) {
        if (t < s) red[t] += red[t + s];
        __syncthreads();
    }
    float total = red[0];
    __syncthreads();

    // Pass 3: normalise.
    for (int i = t; i < cols; i += HB_TILE)
        ro[i] /= total;
}

// The fused pair written competently, so the benchmark has a baseline that can
// win. One block per row, 256 threads, one shared-memory tree reduction for the
// sum of squares, then a strided pass that normalises and rotates together.
//
// This is the kernel the substitution has to beat to be worth making, and it is
// the fair opponent for a fused Triton kernel: it fuses too.

#ifndef HB_SCALAR
#define HB_SCALAR float
#endif

#define HB_TILE 256

__global__ void rms_norm_rope_tuned(const HB_SCALAR *in, const HB_SCALAR *gamma,
                                    const HB_SCALAR *cos_tab, const HB_SCALAR *sin_tab,
                                    HB_SCALAR *out, int rows, int cols) {
    __shared__ float red[HB_TILE];

    int row = blockIdx.x;
    if (row >= rows) return;

    int half = cols / 2;
    const HB_SCALAR *ri = in + (size_t)row * cols;
    HB_SCALAR *ro = out + (size_t)row * cols;
    const HB_SCALAR *rc = cos_tab + (size_t)row * half;
    const HB_SCALAR *rs = sin_tab + (size_t)row * half;
    int t = threadIdx.x;

    float acc = 0.0f;
    for (int i = t; i < cols; i += HB_TILE) {
        float v = (float)ri[i];
        acc += v * v;
    }
    red[t] = acc;
    __syncthreads();
    for (int stride = HB_TILE / 2; stride > 0; stride >>= 1) {
        if (t < stride) red[t] += red[t + stride];
        __syncthreads();
    }
    float inv = rsqrtf(red[0] / (float)cols + 1e-5f);
    __syncthreads();

    for (int i = t; i < half; i += HB_TILE) {
        float y0 = (float)ri[2 * i] * inv * (float)gamma[2 * i];
        float y1 = (float)ri[2 * i + 1] * inv * (float)gamma[2 * i + 1];
        float c = (float)rc[i];
        float s = (float)rs[i];
        ro[2 * i] = (HB_SCALAR)(y0 * c - y1 * s);
        ro[2 * i + 1] = (HB_SCALAR)(y0 * s + y1 * c);
    }
}

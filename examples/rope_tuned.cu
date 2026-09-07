// RoPE with the row spread across a block, so the benchmark has a baseline that
// can win. No reduction to do: each thread takes a stride of pairs and rotates
// them independently, which is the whole point of this kernel shape.
#define HB_TILE 256

__global__ void rope_tuned(const float *in, const float *cos_tab, const float *sin_tab,
                           float *out, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    int half = cols / 2;
    const float *ri = in + (size_t)row * cols;
    float *ro = out + (size_t)row * cols;
    const float *rc = cos_tab + (size_t)row * half;
    const float *rs = sin_tab + (size_t)row * half;

    for (int i = threadIdx.x; i < half; i += HB_TILE) {
        float x0 = ri[2 * i];
        float x1 = ri[2 * i + 1];
        float c = rc[i];
        float s = rs[i];
        ro[2 * i] = x0 * c - x1 * s;
        ro[2 * i + 1] = x0 * s + x1 * c;
    }
}

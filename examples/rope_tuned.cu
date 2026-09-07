// RoPE with the row spread across a block, so the benchmark has a baseline that
// can win. No reduction to do: each thread takes a stride of pairs and rotates
// them independently, which is the whole point of this kernel shape.

// Element type comes from the driver, which defines HB_SCALAR for the run.
// Loads and stores use it; the arithmetic in between is float, because a half
// precision sum of a long row loses most of its mantissa and the point of this
// kernel is to be a fair reference, not a fast one.
#ifndef HB_SCALAR
#define HB_SCALAR float
#endif
#define HB_TILE 256

__global__ void rope_tuned(const HB_SCALAR *in, const HB_SCALAR *cos_tab, const HB_SCALAR *sin_tab,
                           HB_SCALAR *out, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    int half = cols / 2;
    const HB_SCALAR *ri = in + (size_t)row * cols;
    HB_SCALAR *ro = out + (size_t)row * cols;
    const HB_SCALAR *rc = cos_tab + (size_t)row * half;
    const HB_SCALAR *rs = sin_tab + (size_t)row * half;

    for (int i = threadIdx.x; i < half; i += HB_TILE) {
        float x0 = (float)ri[2 * i];
        float x1 = (float)ri[2 * i + 1];
        float c = (float)rc[i];
        float s = (float)rs[i];
        ro[2 * i] = (HB_SCALAR)(x0 * c - x1 * s);
        ro[2 * i + 1] = (HB_SCALAR)(x0 * s + x1 * c);
    }
}

// Rotary position embedding, as it is usually first written.
//
// Not a reduction: every output pair depends only on the matching input pair
// and the position's angle. It is here because it is the third most common
// kernel in a transformer's inference path and because it is the first one in
// this repo whose extra operands are per-position tables rather than per-column
// weights, which is a different shape entirely.
//
// One row is one position, cols is the head dimension, and the tables hold
// cols/2 angles per row. (in, cos_tab, sin_tab, out, rows, cols).

// Element type comes from the driver, which defines HB_SCALAR for the run.
// Loads and stores use it; the arithmetic in between is float, because a half
// precision sum of a long row loses most of its mantissa and the point of this
// kernel is to be a fair reference, not a fast one.
#ifndef HB_SCALAR
#define HB_SCALAR float
#endif
__global__ void rope(const HB_SCALAR *in, const HB_SCALAR *cos_tab, const HB_SCALAR *sin_tab,
                     HB_SCALAR *out, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    int half = cols / 2;
    const HB_SCALAR *ri = in + (size_t)row * cols;
    HB_SCALAR *ro = out + (size_t)row * cols;
    const HB_SCALAR *rc = cos_tab + (size_t)row * half;
    const HB_SCALAR *rs = sin_tab + (size_t)row * half;

    for (int i = 0; i < half; i++) {
        float x0 = (float)ri[2 * i];
        float x1 = (float)ri[2 * i + 1];
        float c = (float)rc[i];
        float s = (float)rs[i];
        ro[2 * i] = (HB_SCALAR)(x0 * c - x1 * s);
        ro[2 * i + 1] = (HB_SCALAR)(x0 * s + x1 * c);
    }
}

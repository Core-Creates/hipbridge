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
__global__ void rope(const float *in, const float *cos_tab, const float *sin_tab,
                     float *out, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    int half = cols / 2;
    const float *ri = in + (size_t)row * cols;
    float *ro = out + (size_t)row * cols;
    const float *rc = cos_tab + (size_t)row * half;
    const float *rs = sin_tab + (size_t)row * half;

    for (int i = 0; i < half; i++) {
        float x0 = ri[2 * i];
        float x1 = ri[2 * i + 1];
        float c = rc[i];
        float s = rs[i];
        ro[2 * i] = x0 * c - x1 * s;
        ro[2 * i + 1] = x0 * s + x1 * c;
    }
}

// RMSNorm followed by rotary embedding, as the pair is usually first written:
// one block per row, one thread, two walks over the row.
//
// This is the shape a transformer's attention path actually has, and writing it
// as one kernel is not an optimisation anybody made on purpose here. It is the
// naive spelling of two operations that happen to be adjacent, which is exactly
// what a substitution tool is handed.
//
// (in, gamma, cos_tab, sin_tab, out, rows, cols)

// Element type comes from the driver, which defines HB_SCALAR for the run.
// Loads and stores use it; the arithmetic in between is float, because a half
// precision sum of a long row loses most of its mantissa and the point of this
// kernel is to be a fair reference, not a fast one.
#ifndef HB_SCALAR
#define HB_SCALAR float
#endif

__global__ void rms_norm_rope(const HB_SCALAR *in, const HB_SCALAR *gamma,
                              const HB_SCALAR *cos_tab, const HB_SCALAR *sin_tab,
                              HB_SCALAR *out, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    int half = cols / 2;
    const HB_SCALAR *ri = in + (size_t)row * cols;
    HB_SCALAR *ro = out + (size_t)row * cols;
    const HB_SCALAR *rc = cos_tab + (size_t)row * half;
    const HB_SCALAR *rs = sin_tab + (size_t)row * half;

    float acc = 0.0f;
    for (int i = 0; i < cols; i++)
        acc += (float)ri[i] * (float)ri[i];

    float inv = rsqrtf(acc / (float)cols + 1e-5f);

    for (int i = 0; i < half; i++) {
        float y0 = (float)ri[2 * i] * inv * (float)gamma[2 * i];
        float y1 = (float)ri[2 * i + 1] * inv * (float)gamma[2 * i + 1];
        float c = (float)rc[i];
        float s = (float)rs[i];
        ro[2 * i] = (HB_SCALAR)(y0 * c - y1 * s);
        ro[2 * i + 1] = (HB_SCALAR)(y0 * s + y1 * c);
    }
}

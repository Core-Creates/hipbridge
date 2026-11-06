// Pure elementwise, computer-vision edition. One output pixel per thread, reads
// three planar input channels (R, G, B), no loops, no shared memory, no
// cross-lane communication. Classifies as elementwise.flat.
//
// Planar (separate R/G/B pointers) rather than interleaved RGB on purpose: it
// keeps every access a unit-stride elementwise load, which is exactly the
// shape the elementwise recognizer claims and nothing more.
//
// Element type comes from the driver, which defines HB_SCALAR for the run. Loads
// AND the output use it: the driver allocates every buffer as HB_SCALAR, so a
// hardcoded `float *out` only compiled when HB_SCALAR happened to be float and
// broke the fp16/bf16 sweep with a pointer-type mismatch. The arithmetic stays
// float, because the Rec.601 luma weights are full precision and a half-precision
// weighted sum would throw away mantissa before the single rounding back to the
// caller's dtype on store.
#ifndef HB_SCALAR
#define HB_SCALAR float
#endif
__global__ void rgb_to_gray(const HB_SCALAR *r, const HB_SCALAR *g,
                            const HB_SCALAR *b, HB_SCALAR *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = (HB_SCALAR)(0.299f * (float)r[i] + 0.587f * (float)g[i] + 0.114f * (float)b[i]);
}

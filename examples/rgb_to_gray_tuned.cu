// The competently hand-written HIP baseline for rgb_to_gray: the version a
// careful engineer would write by hand on AMD, used as the "why not just port
// it yourself" comparison point. Same maths as rgb_to_gray.cu; a 256-thread
// block (a multiple of the 64-wide CDNA wavefront) is the only change, matching
// the tuned-baseline convention the other suites follow.
#ifndef HB_SCALAR
#define HB_SCALAR float
#endif
__global__ void rgb_to_gray_tuned(const HB_SCALAR *r, const HB_SCALAR *g,
                                  const HB_SCALAR *b, HB_SCALAR *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = (HB_SCALAR)(0.299f * (float)r[i] + 0.587f * (float)g[i] + 0.114f * (float)b[i]);
}

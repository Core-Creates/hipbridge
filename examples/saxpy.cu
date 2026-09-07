// Pure elementwise. One output element per thread, no loops, no shared memory,
// no cross-lane communication.

// Element type comes from the driver, which defines HB_SCALAR for the run.
// Loads and stores use it; the arithmetic in between is float, because a half
// precision sum of a long row loses most of its mantissa and the point of this
// kernel is to be a fair reference, not a fast one.
#ifndef HB_SCALAR
#define HB_SCALAR float
#endif
__global__ void saxpy(const HB_SCALAR *x, const HB_SCALAR *y, float *z, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        z[i] = a * x[i] + y[i];
}

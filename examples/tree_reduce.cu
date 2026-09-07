// Shared-memory tree reduction. Canonical CUDA idiom, uses neither atomics
// nor warp shuffles, which is exactly the case a keyword-matching classifier
// misses.

// Element type comes from the driver, which defines HB_SCALAR for the run.
// Loads and stores use it; the arithmetic in between is float, because a half
// precision sum of a long row loses most of its mantissa and the point of this
// kernel is to be a fair reference, not a fast one.
#ifndef HB_SCALAR
#define HB_SCALAR float
#endif
__global__ void tree_reduce(const HB_SCALAR *in, HB_SCALAR *out, int n) {
    __shared__ float tile[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    tile[tid] = (idx < n) ? in[idx] : 0.0f;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset)
            tile[tid] += tile[tid + offset];
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = tile[0];
}

// Shared-memory tree reduction. Canonical CUDA idiom, uses neither atomics
// nor warp shuffles, which is exactly the case a keyword-matching classifier
// misses.
__global__ void tree_reduce(const float *in, float *out, int n) {
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

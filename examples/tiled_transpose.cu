// Shared-memory tiled transpose with bank-conflict padding. There is no
// recognizer for this shape yet, so it must come back UNKNOWN rather than
// being quietly absorbed into some default pattern.
__global__ void tiled_transpose(const float *in, float *out, int w, int h) {
    __shared__ float tile[32][33];

    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    if (x < w && y < h)
        tile[threadIdx.y][threadIdx.x] = in[y * w + x];

    __syncthreads();

    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;

    if (x < h && y < w)
        out[y * h + x] = tile[threadIdx.x][threadIdx.y];
}

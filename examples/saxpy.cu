// Pure elementwise. One output element per thread, no loops, no shared memory,
// no cross-lane communication.
__global__ void saxpy(const float *x, const float *y, float *z, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        z[i] = a * x[i] + y[i];
}

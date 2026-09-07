// RMSNorm with a learned scale, the form most current LLMs use. No mean, no
// shift: one accumulation of squares, one reciprocal square root, one weight.
//
// Argument order follows the generated driver: every input buffer, then the
// output, then the scalars. So (in, gamma, out, rows, cols).
__global__ void rms_norm_affine(const float *in, const float *gamma,
                                float *out, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float *ri = in + (size_t)row * cols;
    float *ro = out + (size_t)row * cols;

    float acc = 0.0f;
    for (int i = 0; i < cols; i++)
        acc += ri[i] * ri[i];

    float inv = rsqrtf(acc / (float)cols + 1e-5f);

    for (int i = 0; i < cols; i++)
        ro[i] = ri[i] * inv * gamma[i];
}

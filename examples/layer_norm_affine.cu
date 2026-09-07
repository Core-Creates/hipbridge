// LayerNorm as a transformer actually uses it: normalise, then scale by a
// learned gamma and shift by a learned beta.
//
// This is the kernel the harness could not verify until it could carry more
// than one input tensor. Proving the normalisation alone and then shipping a
// substitution that also multiplies by weights would be proving one program and
// deploying another.
//
// Argument order matters and is not free: the generated driver passes every
// input buffer, then the output, then the scalars. So (in, gamma, beta, out,
// rows, cols) and nothing else.
__global__ void layer_norm_affine(const float *in, const float *gamma, const float *beta,
                                  float *out, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float *ri = in + (size_t)row * cols;
    float *ro = out + (size_t)row * cols;

    float sum = 0.0f;
    for (int i = 0; i < cols; i++)
        sum += ri[i];
    float mean = sum / (float)cols;

    float acc = 0.0f;
    for (int i = 0; i < cols; i++) {
        float d = ri[i] - mean;
        acc += d * d;
    }
    float inv = rsqrtf(acc / (float)cols + 1e-5f);

    for (int i = 0; i < cols; i++)
        ro[i] = (ri[i] - mean) * inv * gamma[i] + beta[i];
}

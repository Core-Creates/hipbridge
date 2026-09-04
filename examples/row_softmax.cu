// One block per row, three serial passes. Must NOT be classified elementwise:
// dropping the max-subtraction pass is silent on small inputs and catastrophic
// on large logits.
__global__ void row_softmax(const float *in, float *out, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float *ri = in + row * cols;
    float *ro = out + row * cols;

    float m = -1e20f;
    for (int i = 0; i < cols; i++)
        m = fmaxf(m, ri[i]);

    float s = 0.0f;
    for (int i = 0; i < cols; i++) {
        float v = expf(ri[i] - m);
        ro[i] = v;
        s += v;
    }

    for (int i = 0; i < cols; i++)
        ro[i] /= s;
}

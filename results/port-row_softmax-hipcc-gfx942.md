# hipbridge port report

| field | value |
|---|---|
| generated | 2026-09-07 20:04 UTC |
| hipbridge | 0.1.0.dev0 at commit `4167de1` |
| host | 2 (Linux x86_64) |
| device | AMD Radeon Graphics |
| toolchain | hipcc, HIP version: 7.14.60850-0000000 |
| arch | gfx942 |
| torch | 2.9.1+rocm6.4 |

source: `examples/row_softmax.cu`  kernel: `row_softmax`
pattern: `reduce_serial` (likely)
substitute: hipbridge.kernels.softmax (Triton, AMD-tuned)
launch: `block=(1, 1, 1)`

```
PASS  row_softmax vs hipbridge.kernels.softmax (Triton, AMD-tuned): 42/42 cases, worst ulp=32 (max abs 5.960e-08)  [accuracy vs original: equivalent=42, up to 19x closer to float64]
```

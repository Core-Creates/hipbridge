# hipbridge port report

| field | value |
|---|---|
| generated | 2026-09-08 22:56 UTC |
| hipbridge | 0.1.0.dev0 at commit `8c1834e` |
| host | 2 (Linux x86_64) |
| device | AMD Radeon Graphics |
| toolchain | hipcc, HIP version: 7.14.60850-0000000 |
| arch | gfx942 |
| torch | 2.9.1+rocm6.4 |
| measured code | `7aa3e84fc2a962b5` |

source: `examples/row_softmax.cu`  kernel: `row_softmax`
pattern: `reduce_serial` (likely)
substitute: hipbridge.kernels.softmax (Triton, AMD-tuned)
launch: `block=(1, 1, 1)`

```
PASS  row_softmax vs hipbridge.kernels.softmax (Triton, AMD-tuned): 31/31 cases, worst ulp=574 (max abs 2.086e-07)  [accuracy vs original: better=6, equivalent=25, up to 260x closer to float64]
```

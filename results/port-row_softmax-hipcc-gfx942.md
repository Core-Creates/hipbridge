# hipbridge port report

| field | value |
|---|---|
| generated | 2026-09-07 16:54 UTC |
| hipbridge | 0.1.0.dev0 at commit `c99686e` |
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
PASS  row_softmax vs hipbridge.kernels.softmax (Triton, AMD-tuned): 84/84 cases, worst ulp=593 (max abs 1.199e-07)  [accuracy vs original: better=6, equivalent=78, up to 297x closer to float64]
```

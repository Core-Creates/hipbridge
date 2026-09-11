# hipbridge port report

| field | value |
|---|---|
| generated | 2026-09-11 16:05 UTC |
| hipbridge | 0.1.0.dev0 at commit `8806003` |
| host | 2 (Linux x86_64) |
| device | AMD Radeon Graphics |
| toolchain | hipcc, HIP version: 7.14.60850-0000000 |
| arch | gfx942 |
| torch | 2.9.1+rocm6.4 |
| measured code | `c572bb7c221e07b1` |

source: `examples/row_softmax.cu`  kernel: `row_softmax`
pattern: `reduce_serial` (likely)
substitute: hipbridge.kernels.softmax (Triton, AMD-tuned)
launch: `block=(1, 1, 1)`

```
PASS  row_softmax vs hipbridge.kernels.softmax (Triton, AMD-tuned): 31/31 cases, worst ulp=4454 (max abs 2.461e-07)  [accuracy vs original: better=9, equivalent=22, up to 1034x closer to float64]
```

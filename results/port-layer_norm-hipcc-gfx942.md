# hipbridge port report

| field | value |
|---|---|
| generated | 2026-09-08 04:34 UTC |
| hipbridge | 0.1.0.dev0 at commit `13acca4` |
| host | 2 (Linux x86_64) |
| device | AMD Radeon Graphics |
| toolchain | hipcc, HIP version: 7.14.60850-0000000 |
| arch | gfx942 |
| torch | 2.9.1+rocm6.4 |
| measured code | `4736047d7ba708bf` |

source: `examples/layer_norm.cu`  kernel: `layer_norm`
pattern: `reduce_serial` (likely)
substitute: hipbridge.kernels.norm.layer_norm (Triton, AMD-tuned)
launch: `block=(1, 1, 1)`

```
PASS  layer_norm vs hipbridge.kernels.norm.layer_norm (Triton, AMD-tuned): 28/28 cases, worst ulp=878562811 (max abs 4.768e-07)  [accuracy vs original: equivalent=28, up to 2x closer to float64]
```

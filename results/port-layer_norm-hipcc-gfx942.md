# hipbridge port report

| field | value |
|---|---|
| generated | 2026-09-07 00:43 UTC |
| hipbridge | 0.1.0.dev0 at commit `71724a5` |
| host | 2 (Linux x86_64) |
| device | AMD Radeon Graphics |
| toolchain | hipcc, HIP version: 7.14.60850-0000000 |
| arch | gfx942 |
| torch | 2.9.1+rocm6.4 |

source: `examples/layer_norm.cu`  kernel: `layer_norm`
pattern: `reduce_serial` (likely)
substitute: hipbridge.kernels.norm.layer_norm (Triton, AMD-tuned)
launch: `block=(1, 1, 1)`

```
PASS  layer_norm vs hipbridge.kernels.norm.layer_norm (Triton, AMD-tuned): 84/84 cases, worst ulp=1776828265 (max abs 3.910e-05)  [accuracy vs original: better=5, equivalent=79, up to 79x closer to float64]
```

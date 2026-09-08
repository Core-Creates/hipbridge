# hipbridge port report

| field | value |
|---|---|
| generated | 2026-09-07 23:38 UTC |
| hipbridge | 0.1.0.dev0 at commit `c2da609` |
| host | 2 (Linux x86_64) |
| device | AMD Radeon Graphics |
| toolchain | hipcc, HIP version: 7.14.60850-0000000 |
| arch | gfx942 |
| torch | 2.9.1+rocm6.4 |
| measured code | `7503dc7ace15d391` |

source: `examples/layer_norm.cu`  kernel: `layer_norm`
pattern: `reduce_serial` (likely)
substitute: hipbridge.kernels.norm.layer_norm (Triton, AMD-tuned)
launch: `block=(1, 1, 1)`

```
PASS  layer_norm vs hipbridge.kernels.norm.layer_norm (Triton, AMD-tuned): 42/42 cases, worst ulp=886945686 (max abs 1.431e-06)  [accuracy vs original: equivalent=42, up to 5x closer to float64]
```

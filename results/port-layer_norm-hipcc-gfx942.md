# hipbridge port report

| field | value |
|---|---|
| generated | 2026-09-08 22:24 UTC |
| hipbridge | 0.1.0.dev0 at commit `a8ca0a7` |
| host | 2 (Linux x86_64) |
| device | AMD Radeon Graphics |
| toolchain | hipcc, HIP version: 7.14.60850-0000000 |
| arch | gfx942 |
| torch | 2.9.1+rocm6.4 |
| measured code | `6ab57a4fba756495` |

source: `examples/layer_norm.cu`  kernel: `layer_norm`
pattern: `reduce_serial` (likely)
substitute: hipbridge.kernels.norm.layer_norm (Triton, AMD-tuned), eps=1e-05
launch: `block=(1, 1, 1)`

```
PASS  layer_norm vs hipbridge.kernels.norm.layer_norm (Triton, AMD-tuned), eps=1e-05: 28/28 cases, worst ulp=1751849056 (max abs 1.314e-04)  [accuracy vs original: better=8, equivalent=20, up to 695x closer to float64]
```

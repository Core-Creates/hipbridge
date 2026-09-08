# hipbridge port report

| field | value |
|---|---|
| generated | 2026-09-08 05:08 UTC |
| hipbridge | 0.1.0.dev0 at commit `5bcfd99` |
| host | 2 (Linux x86_64) |
| device | AMD Radeon Graphics |
| toolchain | hipcc, HIP version: 7.14.60850-0000000 |
| arch | gfx942 |
| torch | 2.9.1+rocm6.4 |
| measured code | `1b2feaa6b6416e9c` |

source: `examples/rms_norm.cu`  kernel: `rms_norm`
pattern: `reduce_serial` (likely)
substitute: hipbridge.kernels.norm.rms_norm (Triton, AMD-tuned)
launch: `block=(1, 1, 1)`

```
PASS  rms_norm vs hipbridge.kernels.norm.rms_norm (Triton, AMD-tuned): 28/28 cases, worst ulp=12 (max abs 1.907e-06)  [accuracy vs original: equivalent=28, up to 7x closer to float64]
```

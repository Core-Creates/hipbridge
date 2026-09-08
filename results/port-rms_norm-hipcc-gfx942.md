# hipbridge port report

| field | value |
|---|---|
| generated | 2026-09-08 04:35 UTC |
| hipbridge | 0.1.0.dev0 at commit `13acca4` |
| host | 2 (Linux x86_64) |
| device | AMD Radeon Graphics |
| toolchain | hipcc, HIP version: 7.14.60850-0000000 |
| arch | gfx942 |
| torch | 2.9.1+rocm6.4 |
| measured code | `4736047d7ba708bf` |

source: `examples/rms_norm.cu`  kernel: `rms_norm`
pattern: `reduce_serial` (likely)
substitute: hipbridge.kernels.norm.rms_norm (Triton, AMD-tuned)
launch: `block=(1, 1, 1)`

```
PASS  rms_norm vs hipbridge.kernels.norm.rms_norm (Triton, AMD-tuned): 28/28 cases, worst ulp=2 (max abs 2.384e-07)  [accuracy vs original: equivalent=28]
```

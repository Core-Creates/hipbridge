# hipbridge port report

| field | value |
|---|---|
| generated | 2026-09-08 22:57 UTC |
| hipbridge | 0.1.0.dev0 at commit `8c1834e` |
| host | 2 (Linux x86_64) |
| device | AMD Radeon Graphics |
| toolchain | hipcc, HIP version: 7.14.60850-0000000 |
| arch | gfx942 |
| torch | 2.9.1+rocm6.4 |
| measured code | `7aa3e84fc2a962b5` |

source: `examples/rms_norm.cu`  kernel: `rms_norm`
pattern: `reduce_serial` (likely)
substitute: hipbridge.kernels.norm.rms_norm (Triton, AMD-tuned), eps=1e-05
launch: `block=(1, 1, 1)`

```
PASS  rms_norm vs hipbridge.kernels.norm.rms_norm (Triton, AMD-tuned), eps=1e-05: 31/31 cases, worst ulp=12 (max abs 1.907e-06)  [accuracy vs original: equivalent=31]
```

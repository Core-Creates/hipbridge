# hipbridge port report

| field | value |
|---|---|
| generated | 2026-09-07 23:39 UTC |
| hipbridge | 0.1.0.dev0 at commit `c2da609` |
| host | 2 (Linux x86_64) |
| device | AMD Radeon Graphics |
| toolchain | hipcc, HIP version: 7.14.60850-0000000 |
| arch | gfx942 |
| torch | 2.9.1+rocm6.4 |
| measured code | `7503dc7ace15d391` |

source: `examples/rms_norm.cu`  kernel: `rms_norm`
pattern: `reduce_serial` (likely)
substitute: hipbridge.kernels.norm.rms_norm (Triton, AMD-tuned)
launch: `block=(1, 1, 1)`

```
PASS  rms_norm vs hipbridge.kernels.norm.rms_norm (Triton, AMD-tuned): 42/42 cases, worst ulp=2 (max abs 4.768e-07)  [accuracy vs original: equivalent=42]
```

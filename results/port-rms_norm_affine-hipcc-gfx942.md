# hipbridge port report

| field | value |
|---|---|
| generated | 2026-09-07 21:00 UTC |
| hipbridge | 0.1.0.dev0 at commit `e413170` |
| host | 2 (Linux x86_64) |
| device | AMD Radeon Graphics |
| toolchain | hipcc, HIP version: 7.14.60850-0000000 |
| arch | gfx942 |
| torch | 2.9.1+rocm6.4 |

source: `examples/rms_norm_affine.cu`  kernel: `rms_norm_affine`
pattern: `reduce_serial` (likely)
substitute: hipbridge.kernels.norm.rms_norm_affine (Triton, AMD-tuned)
launch: `block=(1, 1, 1)`

```
PASS  rms_norm_affine vs hipbridge.kernels.norm.rms_norm_affine (Triton, AMD-tuned): 42/42 cases, worst ulp=2 (max abs 1.526e-05)  [accuracy vs original: equivalent=42]
```

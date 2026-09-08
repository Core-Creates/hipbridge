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

source: `examples/rms_norm_affine.cu`  kernel: `rms_norm_affine`
pattern: `reduce_serial` (likely)
substitute: hipbridge.kernels.norm.rms_norm_affine (Triton, AMD-tuned)
launch: `block=(1, 1, 1)`

```
PASS  rms_norm_affine vs hipbridge.kernels.norm.rms_norm_affine (Triton, AMD-tuned): 28/28 cases, worst ulp=12 (max abs 2.441e-04)  [accuracy vs original: equivalent=28, up to 5x closer to float64]
```

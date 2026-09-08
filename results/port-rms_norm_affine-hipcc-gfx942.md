# hipbridge port report

| field | value |
|---|---|
| generated | 2026-09-08 22:25 UTC |
| hipbridge | 0.1.0.dev0 at commit `a8ca0a7` |
| host | 2 (Linux x86_64) |
| device | AMD Radeon Graphics |
| toolchain | hipcc, HIP version: 7.14.60850-0000000 |
| arch | gfx942 |
| torch | 2.9.1+rocm6.4 |
| measured code | `6ab57a4fba756495` |

source: `examples/rms_norm_affine.cu`  kernel: `rms_norm_affine`
pattern: `reduce_serial` (likely)
substitute: hipbridge.kernels.norm.rms_norm_affine (Triton, AMD-tuned), eps=1e-05
launch: `block=(1, 1, 1)`

```
PASS  rms_norm_affine vs hipbridge.kernels.norm.rms_norm_affine (Triton, AMD-tuned), eps=1e-05: 28/28 cases, worst ulp=12 (max abs 2.441e-04)  [accuracy vs original: equivalent=28]
```

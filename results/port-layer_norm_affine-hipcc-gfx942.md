# hipbridge port report

| field | value |
|---|---|
| generated | 2026-09-12 03:50 UTC |
| hipbridge | 0.1.0.dev0 at commit `bad9fab` |
| host | 2 (Linux x86_64) |
| device | AMD Radeon Graphics |
| toolchain | hipcc, HIP version: 7.14.60850-0000000 |
| arch | gfx942 |
| torch | 2.9.1+rocm6.4 |
| measured code | `c2d376c8a380e3ee` |

source: `examples/layer_norm_affine.cu`  kernel: `layer_norm_affine`
pattern: `reduce_serial` (likely)
substitute: hipbridge.kernels.norm.layer_norm_affine (Triton, AMD-tuned), eps=1e-05
launch: `block=(1, 1, 1)`

```
PASS  layer_norm_affine vs hipbridge.kernels.norm.layer_norm_affine (Triton, AMD-tuned), eps=1e-05: 31/31 cases, worst ulp=22563152 (max abs 1.404e-03)  [accuracy vs original: better=8, equivalent=23, up to 457x closer to float64]
```

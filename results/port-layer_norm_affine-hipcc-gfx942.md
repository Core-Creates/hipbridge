# hipbridge port report

| field | value |
|---|---|
| generated | 2026-09-08 20:34 UTC |
| hipbridge | 0.1.0.dev0 at commit `b333833` |
| host | 2 (Linux x86_64) |
| device | AMD Radeon Graphics |
| toolchain | hipcc, HIP version: 7.14.60850-0000000 |
| arch | gfx942 |
| torch | 2.9.1+rocm6.4 |
| measured code | `2ecfdefc69200090` |

source: `examples/layer_norm_affine.cu`  kernel: `layer_norm_affine`
pattern: `reduce_serial` (likely)
substitute: hipbridge.kernels.norm.layer_norm_affine (Triton, AMD-tuned), eps=1e-05
launch: `block=(1, 1, 1)`

```
PASS  layer_norm_affine vs hipbridge.kernels.norm.layer_norm_affine (Triton, AMD-tuned), eps=1e-05: 28/28 cases, worst ulp=6567080 (max abs 1.099e-03)  [accuracy vs original: better=7, equivalent=21, up to 427x closer to float64]
```

# hipbridge port report

| field | value |
|---|---|
| generated | 2026-09-11 16:07 UTC |
| hipbridge | 0.1.0.dev0 at commit `8806003` |
| host | 2 (Linux x86_64) |
| device | AMD Radeon Graphics |
| toolchain | hipcc, HIP version: 7.14.60850-0000000 |
| arch | gfx942 |
| torch | 2.9.1+rocm6.4 |
| measured code | `c572bb7c221e07b1` |

source: `examples/rms_norm_rope.cu`  kernel: `rms_norm_rope`
pattern: `reduce_serial` (likely)
substitute: hipbridge.kernels.fused.rms_norm_rope (Triton, fused), eps=1e-05
launch: `block=(1, 1, 1)`

```
PASS  rms_norm_rope vs hipbridge.kernels.fused.rms_norm_rope (Triton, fused), eps=1e-05: 31/31 cases, worst ulp=247803 (max abs 3.662e-04)  [accuracy vs original: better=1, equivalent=30, up to 6x closer to float64]
```

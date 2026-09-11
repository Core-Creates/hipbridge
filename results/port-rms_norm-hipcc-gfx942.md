# hipbridge port report

| field | value |
|---|---|
| generated | 2026-09-11 16:06 UTC |
| hipbridge | 0.1.0.dev0 at commit `8806003` |
| host | 2 (Linux x86_64) |
| device | AMD Radeon Graphics |
| toolchain | hipcc, HIP version: 7.14.60850-0000000 |
| arch | gfx942 |
| torch | 2.9.1+rocm6.4 |
| measured code | `c572bb7c221e07b1` |

source: `examples/rms_norm.cu`  kernel: `rms_norm`
pattern: `reduce_serial` (likely)
substitute: hipbridge.kernels.norm.rms_norm (Triton, AMD-tuned), eps=1e-05
launch: `block=(1, 1, 1)`

```
PASS  rms_norm vs hipbridge.kernels.norm.rms_norm (Triton, AMD-tuned), eps=1e-05: 31/31 cases, worst ulp=60 (max abs 1.431e-05)  [accuracy vs original: better=4, equivalent=27, up to 77x closer to float64]
```

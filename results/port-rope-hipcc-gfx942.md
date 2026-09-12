# hipbridge port report

| field | value |
|---|---|
| generated | 2026-09-12 03:51 UTC |
| hipbridge | 0.1.0.dev0 at commit `bad9fab` |
| host | 2 (Linux x86_64) |
| device | AMD Radeon Graphics |
| toolchain | hipcc, HIP version: 7.14.60850-0000000 |
| arch | gfx942 |
| torch | 2.9.1+rocm6.4 |
| measured code | `c2d376c8a380e3ee` |

source: `examples/rope.cu`  kernel: `rope`
pattern: `row_map` (likely)
substitute: hipbridge.kernels.rope (Triton, AMD-tuned)
launch: `block=(1, 1, 1)`

```
PASS  rope vs hipbridge.kernels.rope (Triton, AMD-tuned): 31/31 cases, worst ulp=0 (max abs 0.000e+00)  [accuracy vs original: equivalent=31]
```

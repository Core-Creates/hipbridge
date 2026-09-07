# hipbridge port report

| field | value |
|---|---|
| generated | 2026-09-07 19:40 UTC |
| hipbridge | 0.1.0.dev0 at commit `89e542b` |
| host | 2 (Linux x86_64) |
| device | no device visible to torch |
| toolchain | hipcc, HIP version: 7.14.60850-0000000 |
| arch | gfx942 |
| torch | 2.14.0+cu130 |

source: `examples/rope.cu`  kernel: `rope`
pattern: `row_map` (likely)
substitute: torch rope (Triton unavailable)
launch: `block=(1, 1, 1)`

```
PASS  rope vs torch rope (Triton unavailable): 42/42 cases, worst ulp=1568 (max abs 3.052e-05)  [accuracy vs original: equivalent=42]
```

# hipbridge verification

| field | value |
|---|---|
| generated | 2026-09-07 00:41 UTC |
| hipbridge | 0.1.0.dev0 at commit `71724a5` |
| host | 2 (Linux x86_64) |
| device | AMD Radeon Graphics |
| toolchain | hipcc, HIP version: 7.14.60850-0000000 |
| arch | gfx942 |
| torch | 2.9.1+rocm6.4 |

```
PASS  row_softmax vs original on hipcc: 84/84 cases, worst ulp=593 (max abs 1.199e-07)  [accuracy vs original: better=6, equivalent=78, up to 297x closer to float64]
```

```
PASS  layer_norm vs original on hipcc: 84/84 cases, worst ulp=1776828265 (max abs 3.910e-05)  [accuracy vs original: better=5, equivalent=79, up to 79x closer to float64]
```

```
PASS  rms_norm vs original on hipcc: 84/84 cases, worst ulp=10 (max abs 1.907e-06)  [accuracy vs original: equivalent=84, up to 15x closer to float64]
```

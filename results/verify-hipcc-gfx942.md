# hipbridge verification

| field | value |
|---|---|
| generated | 2026-09-07 20:01 UTC |
| hipbridge | 0.1.0.dev0 at commit `4167de1` |
| host | 2 (Linux x86_64) |
| device | AMD Radeon Graphics |
| toolchain | hipcc, HIP version: 7.14.60850-0000000 |
| arch | gfx942 |
| torch | 2.9.1+rocm6.4 |

```
PASS  row_softmax vs original on hipcc: 126/126 cases, worst ulp=35 (max abs 4.883e-04)  [accuracy vs original: equivalent=126, up to 16x closer to float64]
```

```
PASS  layer_norm vs original on hipcc: 126/126 cases, worst ulp=886945686 (max abs 2.384e-06)  [accuracy vs original: equivalent=126, up to 27x closer to float64]
```

```
PASS  layer_norm_affine vs original on hipcc: 126/126 cases, worst ulp=397 (max abs 6.104e-05)  [accuracy vs original: equivalent=126, up to 33x closer to float64]
```

```
PASS  rms_norm vs original on hipcc: 126/126 cases, worst ulp=3 (max abs 4.768e-07)  [accuracy vs original: equivalent=126, up to 5x closer to float64]
```

```
PASS  rms_norm_affine vs original on hipcc: 126/126 cases, worst ulp=3 (max abs 3.052e-05)  [accuracy vs original: equivalent=126, up to 2x closer to float64]
```

```
PASS  rope vs original on hipcc: 126/126 cases, worst ulp=0 (max abs 0.000e+00)  [accuracy vs original: equivalent=126]
```

# hipbridge verification

| field | value |
|---|---|
| generated | 2026-09-08 04:30 UTC |
| hipbridge | 0.1.0.dev0 at commit `13acca4` |
| host | 2 (Linux x86_64) |
| device | AMD Radeon Graphics |
| toolchain | hipcc, HIP version: 7.14.60850-0000000 |
| arch | gfx942 |
| torch | 2.9.1+rocm6.4 |
| measured code | `4736047d7ba708bf` |

```
PASS  row_softmax vs original on hipcc: 84/84 cases, worst ulp=31 (max abs 4.883e-04)  [accuracy vs original: equivalent=84, up to 4x closer to float64]
```

```
PASS  layer_norm vs original on hipcc: 84/84 cases, worst ulp=878562811 (max abs 4.768e-07)  [accuracy vs original: equivalent=84, up to 4x closer to float64]
```

```
PASS  layer_norm_affine vs original on hipcc: 84/84 cases, worst ulp=397 (max abs 2.098e-05)  [accuracy vs original: equivalent=84, up to 8x closer to float64]
```

```
PASS  rms_norm vs original on hipcc: 84/84 cases, worst ulp=3 (max abs 4.768e-07)  [accuracy vs original: equivalent=84, up to 5x closer to float64]
```

```
PASS  rms_norm_affine vs original on hipcc: 84/84 cases, worst ulp=3 (max abs 3.576e-07)  [accuracy vs original: equivalent=84, up to 3x closer to float64]
```

```
PASS  rope vs original on hipcc: 84/84 cases, worst ulp=0 (max abs 0.000e+00)  [accuracy vs original: equivalent=84]
```

```
PASS  rms_norm_rope vs original on hipcc: 84/84 cases, worst ulp=69 (max abs 3.052e-05)  [accuracy vs original: equivalent=84, up to 3x closer to float64]
```

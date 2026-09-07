# hipbridge verification

| field | value |
|---|---|
| generated | 2026-09-07 19:37 UTC |
| hipbridge | 0.1.0.dev0 at commit `89e542b` |
| host | 2 (Linux x86_64) |
| device | no device visible to torch |
| toolchain | hipcc, HIP version: 7.14.60850-0000000 |
| arch | gfx942 |
| torch | 2.14.0+cu130 |

```
PASS  row_softmax vs original on hipcc: 126/126 cases, worst ulp=11 (max abs 4.883e-04)  [accuracy vs original: equivalent=126, up to 12x closer to float64]
```

```
PASS  layer_norm vs original on hipcc: 126/126 cases, worst ulp=1757114155 (max abs 6.104e-04)  [accuracy vs original: equivalent=126, up to 17x closer to float64]
```

```
PASS  layer_norm_affine vs original on hipcc: 126/126 cases, worst ulp=423 (max abs 6.250e-02)  [accuracy vs original: equivalent=126, up to 33x closer to float64]
```

```
PASS  rms_norm vs original on hipcc: 126/126 cases, worst ulp=3 (max abs 4.768e-07)  [accuracy vs original: equivalent=126, up to 25x closer to float64]
```

```
PASS  rms_norm_affine vs original on hipcc: 126/126 cases, worst ulp=3 (max abs 3.052e-05)  [accuracy vs original: equivalent=126, up to 5x closer to float64]
```

```
PASS  rope vs original on hipcc: 126/126 cases, worst ulp=16016 (max abs 2.000e+00)  [accuracy vs original: equivalent=126]
```

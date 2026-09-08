# hipbridge verification

| field | value |
|---|---|
| generated | 2026-09-08 05:03 UTC |
| hipbridge | 0.1.0.dev0 at commit `5bcfd99` |
| host | 2 (Linux x86_64) |
| device | AMD Radeon Graphics |
| toolchain | hipcc, HIP version: 7.14.60850-0000000 |
| arch | gfx942 |
| torch | 2.9.1+rocm6.4 |
| measured code | `1b2feaa6b6416e9c` |

```
PASS  row_softmax vs original on hipcc: 84/84 cases, worst ulp=747 (max abs 1.953e-03)  [accuracy vs original: better=4, equivalent=80, up to 201x closer to float64]
```

```
PASS  layer_norm vs original on hipcc: 84/84 cases, worst ulp=1751849056 (max abs 7.812e-03)  [accuracy vs original: better=6, equivalent=78, up to 695x closer to float64]
```

```
PASS  layer_norm_affine vs original on hipcc: 84/84 cases, worst ulp=6567080 (max abs 5.000e-01)  [accuracy vs original: better=6, equivalent=78, up to 427x closer to float64]
```

```
PASS  rms_norm vs original on hipcc: 84/84 cases, worst ulp=14 (max abs 1.953e-03)  [accuracy vs original: equivalent=84, up to 16x closer to float64]
```

```
PASS  rms_norm_affine vs original on hipcc: 84/84 cases, worst ulp=15 (max abs 2.000e+00)  [accuracy vs original: equivalent=84, up to 12x closer to float64]
```

```
PASS  rope vs original on hipcc: 84/84 cases, worst ulp=0 (max abs 0.000e+00)  [accuracy vs original: equivalent=84]
```

```
PASS  rms_norm_rope vs original on hipcc: 84/84 cases, worst ulp=247803 (max abs 2.000e+00)  [accuracy vs original: equivalent=84, up to 6x closer to float64]
```

# hipbridge verification

| field | value |
|---|---|
| generated | 2026-09-11 16:01 UTC |
| hipbridge | 0.1.0.dev0 at commit `8806003` |
| host | 2 (Linux x86_64) |
| device | AMD Radeon Graphics |
| toolchain | hipcc, HIP version: 7.14.60850-0000000 |
| arch | gfx942 |
| torch | 2.9.1+rocm6.4 |
| measured code | `c572bb7c221e07b1` |

```
PASS  row_softmax vs original on hipcc: 93/93 cases, worst ulp=5719 (max abs 1.953e-03)  [accuracy vs original: better=9, equivalent=84, up to 3169x closer to float64]
```

```
PASS  layer_norm vs original on hipcc: 93/93 cases, worst ulp=1751849056 (max abs 1.562e-02)  [accuracy vs original: better=12, equivalent=81, up to 440x closer to float64]
```

```
PASS  layer_norm_affine vs original on hipcc: 93/93 cases, worst ulp=27780685 (max abs 4.000e+00)  [accuracy vs original: better=11, equivalent=82, up to 226x closer to float64]
```

```
PASS  rms_norm vs original on hipcc: 93/93 cases, worst ulp=60 (max abs 3.125e-02)  [accuracy vs original: better=6, equivalent=87, up to 51x closer to float64]
```

```
PASS  rms_norm_affine vs original on hipcc: 93/93 cases, worst ulp=62 (max abs 4.000e+00)  [accuracy vs original: better=6, equivalent=87, up to 32x closer to float64]
```

```
PASS  rope vs original on hipcc: 93/93 cases, worst ulp=1 (max abs 4.883e-04)  [accuracy vs original: equivalent=93]
```

```
PASS  rms_norm_rope vs original on hipcc: 93/93 cases, worst ulp=247803 (max abs 1.000e+00)  [accuracy vs original: better=1, equivalent=92, up to 6x closer to float64]
```

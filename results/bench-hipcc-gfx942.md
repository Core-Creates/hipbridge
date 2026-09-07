# hipbridge benchmark

| field | value |
|---|---|
| generated | 2026-09-07 00:42 UTC |
| hipbridge | 0.1.0.dev0 at commit `71724a5` |
| host | 2 (Linux x86_64) |
| device | AMD Radeon Graphics |
| toolchain | hipcc, HIP version: 7.14.60850-0000000 |
| arch | gfx942 |
| torch | 2.9.1+rocm6.4 |

## row_softmax

candidate: hipbridge.kernels.softmax (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       323.9 [323.8-324.0]           3.9 [3.9-4.0]           5.1 [4.8-5.2]        15.9 [15.7-16.1]        20.4x         0.2x         0.3x  (latency-bound)
   1024x1024       364.1 [363.6-364.7]           5.6 [5.6-5.6]           5.8 [5.6-6.0]        18.6 [16.1-19.0]        19.6x         0.3x         0.3x  (latency-bound)
   4096x4096    2430.9 [2430.5-2431.9]        94.2 [92.9-94.3]        40.0 [39.6-45.9]        32.5 [32.4-32.6]        74.9x         2.9x         1.2x
```

## layer_norm

candidate: hipbridge.kernels.norm.layer_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       204.9 [204.8-204.9]           3.9 [3.8-4.0]           8.1 [8.1-8.3]        16.6 [16.4-16.6]        12.4x         0.2x         0.5x  (latency-bound)
   1024x1024       233.5 [233.5-233.6]           5.0 [5.0-5.0]           8.2 [8.1-8.6]        16.4 [16.4-16.6]        14.2x         0.3x         0.5x  (latency-bound)
   4096x4096    1832.3 [1811.7-1846.0]        50.7 [50.5-50.9]        44.7 [44.6-55.3]        31.2 [31.1-31.2]        58.8x         1.6x         1.4x
```

## rms_norm

candidate: hipbridge.kernels.norm.rms_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       169.7 [169.6-169.8]           2.8 [2.8-2.8]           6.9 [6.8-7.0]        16.7 [16.3-16.7]        10.2x         0.2x         0.4x  (latency-bound)
   1024x1024       193.5 [193.4-193.6]           4.0 [4.0-4.1]           7.4 [7.3-7.7]        16.4 [16.4-16.7]        11.8x         0.2x         0.5x  (latency-bound)
   4096x4096    1520.0 [1512.6-1528.3]        41.4 [41.3-41.5]        43.0 [42.8-53.0]        31.4 [31.3-31.4]        48.5x         1.3x         1.4x
```

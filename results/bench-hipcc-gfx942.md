# hipbridge benchmark

| field | value |
|---|---|
| generated | 2026-09-07 23:37 UTC |
| hipbridge | 0.1.0.dev0 at commit `c2da609` |
| host | 2 (Linux x86_64) |
| device | AMD Radeon Graphics |
| toolchain | hipcc, HIP version: 7.14.60850-0000000 |
| arch | gfx942 |
| torch | 2.9.1+rocm6.4 |
| measured code | `7503dc7ace15d391` |

## row_softmax

candidate: hipbridge.kernels.softmax (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       326.1 [326.0-326.1]           4.0 [4.0-4.0]           5.0 [4.8-5.2]        15.9 [15.8-15.9]        20.5x         0.3x         0.3x  (latency-bound)
   1024x1024       363.9 [363.9-364.4]           6.3 [6.3-6.4]           5.3 [5.2-5.3]        16.0 [15.8-16.1]        22.8x         0.4x         0.3x  (latency-bound)
   4096x4096    2431.8 [2430.3-2432.8]        52.9 [51.9-53.1]        40.9 [40.0-41.6]        31.9 [31.7-32.0]        76.1x         1.7x         1.3x
```

## layer_norm

candidate: hipbridge.kernels.norm.layer_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       205.0 [204.9-205.2]           3.9 [3.8-4.0]           8.0 [8.0-8.1]        18.5 [17.8-19.0]        11.1x         0.2x         0.4x  (latency-bound)
   1024x1024       233.5 [233.4-233.6]           5.4 [5.4-5.4]           8.9 [8.4-9.2]        18.5 [17.7-18.6]        12.6x         0.3x         0.5x  (latency-bound)
   4096x4096    1749.2 [1745.2-1751.7]        46.0 [45.8-46.2]        44.7 [44.6-45.2]        30.9 [30.9-31.0]        56.5x         1.5x         1.4x
```

## layer_norm_affine

candidate: hipbridge.kernels.norm.layer_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       209.0 [208.8-209.0]           4.0 [4.0-4.1]           8.2 [8.0-8.4]        19.7 [19.4-19.9]        10.6x         0.2x         0.4x  (latency-bound)
   1024x1024       240.9 [240.8-241.2]           5.5 [5.5-5.5]           8.3 [8.2-8.4]        19.3 [19.2-19.8]        12.5x         0.3x         0.4x  (latency-bound)
   4096x4096    1877.1 [1840.2-1878.3]        47.1 [46.3-47.6]        45.3 [45.0-45.7]        33.5 [33.4-33.5]        56.0x         1.4x         1.4x
```

## rms_norm

candidate: hipbridge.kernels.norm.rms_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       169.9 [169.8-169.9]           2.7 [2.7-2.8]           8.6 [8.5-8.8]        20.0 [19.9-20.7]         8.5x         0.1x         0.4x  (latency-bound)
   1024x1024       193.5 [193.3-193.6]           4.0 [4.0-4.1]           7.7 [7.7-7.7]        18.0 [17.9-18.4]        10.8x         0.2x         0.4x  (latency-bound)
   4096x4096    1460.8 [1448.0-1524.6]        41.3 [41.2-41.7]        43.6 [43.1-43.6]        31.3 [31.2-31.4]        46.6x         1.3x         1.4x
```

## rms_norm_affine

candidate: hipbridge.kernels.norm.rms_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       173.8 [173.7-173.8]           2.8 [2.8-2.8]           7.0 [7.0-7.1]        17.5 [17.5-17.7]         9.9x         0.2x         0.4x  (latency-bound)
   1024x1024       200.3 [200.3-200.5]           4.0 [4.0-4.0]           7.3 [7.2-7.5]        18.7 [18.3-19.8]        10.7x         0.2x         0.4x  (latency-bound)
   4096x4096    1580.0 [1571.0-1645.0]        41.1 [40.9-41.2]        43.8 [43.4-43.9]        32.1 [32.0-32.2]        49.2x         1.3x         1.4x
```

## rope

candidate: hipbridge.kernels.rope (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024          80.4 [80.4-80.4]           2.2 [2.2-2.3]        47.1 [46.0-48.3]        20.2 [20.1-20.2]         4.0x         0.1x         2.3x  (latency-bound)
   1024x1024       101.9 [101.6-102.0]           2.8 [2.8-2.8]        47.6 [47.1-48.4]        21.4 [19.7-21.8]         4.8x         0.1x         2.2x  (latency-bound)
   4096x4096    1371.2 [1370.0-1372.1]        45.9 [45.7-45.9]     241.2 [240.6-242.4]        51.8 [51.6-51.8]        26.5x         0.9x         4.7x
```

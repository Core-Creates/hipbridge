# hipbridge benchmark

| field | value |
|---|---|
| generated | 2026-09-07 16:46 UTC |
| hipbridge | 0.1.0.dev0 at commit `c99686e` |
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
      1x1024       324.3 [321.9-324.7]           4.0 [3.9-4.0]           5.2 [4.8-5.3]        16.5 [16.0-16.6]        19.6x         0.2x         0.3x  (latency-bound)
   1024x1024       380.9 [380.8-382.1]           5.6 [5.6-5.6]           5.3 [5.2-5.8]        15.8 [15.7-16.4]        24.1x         0.4x         0.3x  (latency-bound)
   4096x4096    2429.5 [2426.8-2430.5]        95.1 [94.9-95.4]        40.3 [39.8-46.9]        31.2 [31.2-31.6]        77.8x         3.0x         1.3x
```

## layer_norm

candidate: hipbridge.kernels.norm.layer_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       205.0 [204.8-205.0]           3.9 [3.9-4.1]           8.5 [8.5-8.5]        16.4 [16.3-16.5]        12.5x         0.2x         0.5x  (latency-bound)
   1024x1024       233.4 [233.3-233.6]           5.0 [5.0-5.0]           8.3 [8.2-8.8]        16.1 [16.0-16.2]        14.5x         0.3x         0.5x  (latency-bound)
   4096x4096    1749.5 [1740.5-1749.7]        50.7 [50.5-50.7]        44.6 [44.5-56.2]        30.6 [30.6-30.9]        57.1x         1.7x         1.5x
```

## layer_norm_affine

candidate: hipbridge.kernels.norm.layer_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       208.8 [208.6-208.8]           4.0 [3.9-4.0]           8.2 [8.0-8.2]        19.1 [19.0-20.2]        11.0x         0.2x         0.4x  (latency-bound)
   1024x1024       241.0 [240.8-241.2]           5.1 [5.1-5.1]           9.2 [8.6-9.4]        21.2 [20.8-21.5]        11.4x         0.2x         0.4x  (latency-bound)
   4096x4096    1838.2 [1832.7-1841.7]        51.5 [51.4-51.8]        45.2 [45.1-53.1]        32.8 [32.7-40.3]        56.1x         1.6x         1.4x
```

## rms_norm

candidate: hipbridge.kernels.norm.rms_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       169.6 [169.5-169.7]           2.8 [2.8-2.9]           7.0 [7.0-7.0]        16.0 [16.0-16.1]        10.6x         0.2x         0.4x  (latency-bound)
   1024x1024       193.8 [193.6-193.9]           4.0 [4.0-4.0]           7.2 [7.0-7.5]        16.2 [16.1-16.2]        12.0x         0.2x         0.4x  (latency-bound)
   4096x4096    1465.8 [1462.9-1468.7]        41.4 [41.3-41.4]        43.0 [42.7-52.4]        30.9 [30.9-30.9]        47.4x         1.3x         1.4x
```

## rms_norm_affine

candidate: hipbridge.kernels.norm.rms_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       173.8 [173.7-173.8]           2.8 [2.8-2.8]           7.2 [7.1-7.3]        17.4 [17.4-17.7]        10.0x         0.2x         0.4x  (latency-bound)
   1024x1024       200.5 [200.2-200.6]           4.0 [4.0-4.0]           8.1 [7.8-8.1]        19.4 [19.1-19.4]        10.3x         0.2x         0.4x  (latency-bound)
   4096x4096    1584.5 [1577.4-1587.5]        41.7 [41.7-41.8]        43.3 [43.3-55.3]        31.8 [31.5-31.8]        49.9x         1.3x         1.4x
```

## rope

candidate: hipbridge.kernels.rope (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024          80.5 [80.5-80.5]           2.2 [2.1-2.2]        47.3 [46.7-47.3]        19.7 [19.5-19.8]         4.1x         0.1x         2.4x  (latency-bound)
   1024x1024       102.1 [101.9-102.1]           2.8 [2.8-2.8]        46.6 [46.5-47.7]        19.7 [19.6-19.8]         5.2x         0.1x         2.4x  (latency-bound)
   4096x4096    1372.1 [1371.9-1373.2]        45.8 [45.5-46.0]     242.4 [241.7-247.0]        52.4 [52.3-52.5]        26.2x         0.9x         4.6x
```

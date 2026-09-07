# hipbridge benchmark

| field | value |
|---|---|
| generated | 2026-09-07 20:58 UTC |
| hipbridge | 0.1.0.dev0 at commit `e413170` |
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
      1x1024       323.8 [322.1-324.2]           4.0 [4.0-4.1]           5.2 [4.8-5.3]        15.9 [15.8-16.1]        20.4x         0.3x         0.3x  (latency-bound)
   1024x1024       380.4 [380.2-380.5]           6.3 [6.3-6.4]           5.2 [5.2-5.4]        16.1 [15.8-16.4]        23.7x         0.4x         0.3x  (latency-bound)
   4096x4096    2431.9 [2431.7-2431.9]        53.3 [52.3-53.5]        40.6 [39.9-41.3]        31.7 [31.5-31.8]        76.7x         1.7x         1.3x
```

## layer_norm

candidate: hipbridge.kernels.norm.layer_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       204.9 [204.8-205.1]           4.0 [3.9-4.1]           8.1 [8.0-8.3]        16.5 [16.5-16.6]        12.4x         0.2x         0.5x  (latency-bound)
   1024x1024       233.6 [233.5-233.6]           5.4 [5.4-5.4]           9.1 [9.0-9.1]        17.0 [16.5-17.9]        13.7x         0.3x         0.5x  (latency-bound)
   4096x4096    1739.7 [1738.3-1750.7]        45.7 [45.4-45.7]        44.7 [44.6-44.7]        31.0 [31.0-31.1]        56.1x         1.5x         1.4x
```

## layer_norm_affine

candidate: hipbridge.kernels.norm.layer_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       208.9 [208.7-208.9]           3.9 [3.9-4.0]           8.2 [8.1-8.3]        19.4 [19.2-20.1]        10.7x         0.2x         0.4x  (latency-bound)
   1024x1024       241.0 [241.0-241.1]           5.5 [5.5-5.5]           8.4 [8.4-8.6]        19.2 [19.0-19.2]        12.5x         0.3x         0.4x  (latency-bound)
   4096x4096    1829.9 [1828.8-1847.0]        46.9 [46.6-47.0]        45.7 [45.2-46.2]        33.2 [33.1-33.2]        55.2x         1.4x         1.4x
```

## rms_norm

candidate: hipbridge.kernels.norm.rms_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       169.9 [169.8-169.9]           2.8 [2.8-2.9]           8.1 [8.0-8.1]        17.9 [17.9-18.5]         9.5x         0.2x         0.5x  (latency-bound)
   1024x1024       193.8 [193.7-193.8]           4.0 [4.0-4.0]           6.9 [6.9-7.0]        16.7 [16.6-17.0]        11.6x         0.2x         0.4x  (latency-bound)
   4096x4096    1428.3 [1425.5-1436.1]        41.2 [41.1-41.5]        43.1 [43.0-43.2]        31.2 [31.1-31.5]        45.8x         1.3x         1.4x
```

## rms_norm_affine

candidate: hipbridge.kernels.norm.rms_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       173.9 [173.6-174.0]           2.8 [2.8-2.8]           7.0 [6.9-7.1]        17.6 [17.6-17.8]         9.9x         0.2x         0.4x  (latency-bound)
   1024x1024       200.4 [200.1-200.5]           4.0 [4.0-4.1]           7.1 [6.9-7.2]        17.8 [17.6-18.0]        11.2x         0.2x         0.4x  (latency-bound)
   4096x4096    1578.2 [1578.0-1591.9]        41.4 [41.2-41.4]        43.6 [43.5-43.6]        32.0 [31.6-32.0]        49.4x         1.3x         1.4x
```

## rope

candidate: hipbridge.kernels.rope (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024          80.4 [80.3-80.4]           2.1 [2.1-2.2]        47.1 [46.5-47.6]        19.7 [19.7-20.1]         4.1x         0.1x         2.4x  (latency-bound)
   1024x1024       102.0 [102.0-102.0]           2.8 [2.8-2.8]        46.1 [45.9-46.6]        19.9 [19.7-20.0]         5.1x         0.1x         2.3x  (latency-bound)
   4096x4096    1371.6 [1370.8-1371.9]        45.7 [45.4-45.8]     241.3 [240.3-242.7]        53.1 [53.0-53.5]        25.8x         0.9x         4.5x
```

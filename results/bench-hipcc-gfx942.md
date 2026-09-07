# hipbridge benchmark

| field | value |
|---|---|
| generated | 2026-09-07 20:04 UTC |
| hipbridge | 0.1.0.dev0 at commit `4167de1` |
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
      1x1024       323.9 [323.8-323.9]           4.0 [4.0-4.0]           5.0 [4.8-5.3]        16.0 [15.9-16.4]        20.3x         0.2x         0.3x  (latency-bound)
   1024x1024       384.6 [384.4-384.8]           5.6 [5.6-5.6]           5.2 [5.2-5.5]        16.1 [16.1-16.2]        23.8x         0.3x         0.3x  (latency-bound)
   4096x4096    2430.8 [2430.1-2431.4]        93.9 [93.1-94.3]        38.8 [38.3-40.2]        32.3 [32.2-32.4]        75.3x         2.9x         1.2x
```

## layer_norm

candidate: hipbridge.kernels.norm.layer_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       205.0 [204.9-205.1]           4.0 [3.9-4.0]           8.0 [8.0-8.2]        16.5 [16.4-16.7]        12.4x         0.2x         0.5x  (latency-bound)
   1024x1024       233.6 [233.5-233.6]           5.0 [5.0-5.0]           8.2 [8.2-8.5]        16.5 [16.4-16.7]        14.1x         0.3x         0.5x  (latency-bound)
   4096x4096    1743.2 [1742.8-1753.1]        50.8 [50.7-51.0]        44.6 [44.5-44.6]        31.2 [31.1-31.3]        55.8x         1.6x         1.4x
```

## layer_norm_affine

candidate: hipbridge.kernels.norm.layer_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       208.9 [208.8-208.9]           4.0 [3.9-4.4]           8.4 [8.2-8.5]        19.5 [19.5-19.7]        10.7x         0.2x         0.4x  (latency-bound)
   1024x1024       240.9 [240.7-241.2]           5.1 [5.1-5.1]           8.4 [8.3-8.7]        19.8 [19.5-20.9]        12.2x         0.3x         0.4x  (latency-bound)
   4096x4096    1850.3 [1841.5-1856.6]        51.2 [51.1-51.7]        44.8 [44.6-45.4]        32.9 [32.8-32.9]        56.3x         1.6x         1.4x
```

## rms_norm

candidate: hipbridge.kernels.norm.rms_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       169.8 [169.7-169.9]           2.8 [2.8-2.8]           7.5 [7.2-7.6]        16.6 [16.5-17.7]        10.2x         0.2x         0.5x  (latency-bound)
   1024x1024       193.7 [193.6-193.7]           4.0 [4.0-4.0]           7.8 [7.6-8.1]        17.4 [17.2-17.4]        11.2x         0.2x         0.4x  (latency-bound)
   4096x4096    1453.6 [1443.2-1457.9]        41.2 [41.2-41.2]        43.3 [43.3-43.5]        31.4 [31.2-31.6]        46.3x         1.3x         1.4x
```

## rms_norm_affine

candidate: hipbridge.kernels.norm.rms_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       173.8 [173.7-173.8]           2.9 [2.9-2.9]           7.5 [7.3-7.8]        18.1 [17.8-18.3]         9.6x         0.2x         0.4x  (latency-bound)
   1024x1024       200.5 [200.3-200.5]           4.0 [4.0-4.0]           7.5 [7.4-8.1]        18.4 [18.0-18.5]        10.9x         0.2x         0.4x  (latency-bound)
   4096x4096    1579.5 [1571.3-1580.5]        41.3 [41.2-41.4]        43.3 [43.2-43.8]        31.6 [31.4-31.8]        49.9x         1.3x         1.4x
```

## rope

candidate: hipbridge.kernels.rope (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024          80.4 [80.4-80.4]           2.2 [2.2-2.4]        48.4 [47.1-49.4]        21.3 [19.9-22.4]         3.8x         0.1x         2.3x  (latency-bound)
   1024x1024       101.9 [101.9-102.1]           2.8 [2.8-2.8]        50.1 [49.0-51.2]        20.3 [20.2-20.3]         5.0x         0.1x         2.5x  (latency-bound)
   4096x4096    1374.8 [1374.1-1375.0]        45.6 [45.5-45.9]     239.5 [238.7-240.2]        51.3 [51.2-51.4]        26.8x         0.9x         4.7x
```

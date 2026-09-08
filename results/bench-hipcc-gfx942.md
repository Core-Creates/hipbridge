# hipbridge benchmark

| field | value |
|---|---|
| generated | 2026-09-08 03:45 UTC |
| hipbridge | 0.1.0.dev0 at commit `89999f6` |
| host | 2 (Linux x86_64) |
| device | AMD Radeon Graphics |
| toolchain | hipcc, HIP version: 7.14.60850-0000000 |
| arch | gfx942 |
| torch | 2.9.1+rocm6.4 |
| measured code | `7503dc7ace15d391` |

## row_softmax (float32)

candidate: hipbridge.kernels.softmax (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       324.3 [324.3-324.5]           4.0 [4.0-4.0]           5.2 [4.9-5.2]        16.2 [15.9-16.2]        20.0x         0.2x         0.3x  (latency-bound)
   1024x1024       383.9 [383.4-385.5]           6.3 [6.3-6.3]           5.3 [5.3-5.3]        15.9 [15.8-15.9]        24.2x         0.4x         0.3x  (latency-bound)
   4096x4096    2425.2 [2424.8-2426.4]        53.0 [51.5-53.4]        39.6 [39.2-40.3]        31.3 [31.0-31.4]        77.5x         1.7x         1.3x
```

## row_softmax (float16)

candidate: hipbridge.kernels.softmax (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       309.2 [309.2-309.3]           4.0 [4.0-4.0]           5.5 [5.4-5.7]        16.2 [16.0-16.5]        19.0x         0.2x         0.3x  (latency-bound)
   1024x1024       358.3 [353.0-359.8]           6.4 [6.4-6.4]           6.7 [6.7-6.8]        16.1 [16.1-16.5]        22.2x         0.4x         0.4x  (latency-bound)
   4096x4096    2735.9 [2735.4-2736.3]        44.5 [44.3-45.9]        24.2 [23.7-25.4]        24.0 [23.7-24.1]       113.9x         1.9x         1.0x
```

## row_softmax (bfloat16)

candidate: hipbridge.kernels.softmax (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       340.4 [340.1-340.5]           4.2 [4.1-4.3]           5.6 [5.5-5.7]        16.2 [15.9-16.2]        21.1x         0.3x         0.3x  (latency-bound)
   1024x1024       372.4 [372.2-372.8]           6.5 [6.5-6.7]           7.2 [7.2-7.3]        16.2 [16.2-16.2]        23.0x         0.4x         0.4x  (latency-bound)
   4096x4096    3220.9 [3220.5-3223.2]        46.2 [46.0-46.7]        26.1 [25.1-26.7]        25.5 [25.4-25.8]       126.4x         1.8x         1.0x
```

## layer_norm (float32)

candidate: hipbridge.kernels.norm.layer_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       205.0 [205.0-205.1]           3.9 [3.8-3.9]           8.4 [8.1-8.5]        16.2 [16.1-16.3]        12.6x         0.2x         0.5x  (latency-bound)
   1024x1024       233.7 [233.4-233.7]           5.4 [5.4-5.4]           8.5 [8.3-8.6]        16.4 [16.3-16.7]        14.2x         0.3x         0.5x  (latency-bound)
   4096x4096    1839.9 [1826.1-1855.9]        45.9 [45.9-46.0]        45.5 [45.4-45.8]        30.7 [30.6-30.8]        59.9x         1.5x         1.5x
```

## layer_norm (float16)

candidate: hipbridge.kernels.norm.layer_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       272.1 [271.8-272.1]           4.0 [3.9-4.0]           9.1 [8.7-9.3]        18.5 [18.0-19.3]        14.7x         0.2x         0.5x  (latency-bound)
   1024x1024       297.4 [297.2-297.4]           5.6 [5.6-5.6]           8.5 [8.5-8.9]        16.5 [16.5-16.7]        18.0x         0.3x         0.5x  (latency-bound)
   4096x4096    1364.0 [1363.9-1365.4]        30.5 [30.3-30.6]        28.0 [27.4-30.3]        17.9 [17.7-17.9]        76.2x         1.7x         1.6x
```

## layer_norm (bfloat16)

candidate: hipbridge.kernels.norm.layer_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       291.4 [291.4-291.5]           4.0 [3.9-4.0]           9.0 [8.7-9.0]        17.1 [16.8-17.2]        17.0x         0.2x         0.5x  (latency-bound)
   1024x1024       317.2 [317.0-317.3]           5.6 [5.6-5.6]        10.2 [10.0-11.1]        18.0 [17.0-18.2]        17.7x         0.3x         0.6x  (latency-bound)
   4096x4096    1483.8 [1483.1-1484.1]        31.7 [31.4-32.0]        28.9 [28.6-30.9]        18.5 [18.5-19.5]        80.2x         1.7x         1.6x
```

## layer_norm_affine (float32)

candidate: hipbridge.kernels.norm.layer_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       208.9 [208.8-209.1]           3.9 [3.9-3.9]           8.6 [8.4-8.7]        19.3 [19.2-19.6]        10.8x         0.2x         0.4x  (latency-bound)
   1024x1024       241.0 [241.0-241.3]           5.5 [5.5-5.5]           8.8 [8.5-9.0]        20.1 [19.5-89.1]        12.0x         0.3x         0.4x  (latency-bound)
   4096x4096    1849.7 [1846.3-1883.7]        46.9 [46.8-48.5]        45.7 [45.3-45.9]        32.8 [32.6-33.1]        56.5x         1.4x         1.4x
```

## layer_norm_affine (float16)

candidate: hipbridge.kernels.norm.layer_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       275.8 [275.7-276.0]           4.0 [3.9-4.0]           8.9 [8.7-9.0]        19.3 [19.3-19.3]        14.3x         0.2x         0.5x  (latency-bound)
   1024x1024       305.9 [305.8-306.0]           5.7 [5.7-5.7]           8.9 [8.8-9.4]        19.6 [19.6-21.2]        15.6x         0.3x         0.5x  (latency-bound)
   4096x4096    1424.4 [1423.7-1424.5]        34.1 [33.5-34.6]        30.2 [29.9-31.6]        19.5 [19.4-19.6]        72.9x         1.7x         1.5x
```

## layer_norm_affine (bfloat16)

candidate: hipbridge.kernels.norm.layer_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       301.4 [301.3-301.4]           4.0 [4.0-4.0]           9.7 [9.2-9.8]        21.4 [21.3-21.4]        14.1x         0.2x         0.5x  (latency-bound)
   1024x1024       332.5 [331.8-332.6]           5.8 [5.7-6.0]           8.6 [8.5-8.8]        19.2 [19.1-20.4]        17.3x         0.3x         0.4x  (latency-bound)
   4096x4096    1580.8 [1580.1-1581.4]        35.5 [35.4-35.6]        31.2 [30.9-33.0]        20.5 [20.3-20.6]        77.1x         1.7x         1.5x
```

## rms_norm (float32)

candidate: hipbridge.kernels.norm.rms_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       169.7 [169.6-169.7]           2.8 [2.8-2.8]           7.2 [7.1-7.5]        16.4 [16.2-16.6]        10.3x         0.2x         0.4x  (latency-bound)
   1024x1024       193.6 [193.5-193.8]           4.1 [4.0-4.1]           7.4 [7.2-8.0]        16.5 [16.4-16.7]        11.7x         0.2x         0.4x  (latency-bound)
   4096x4096    1539.2 [1530.1-1546.9]        41.3 [41.1-41.3]        43.7 [43.7-43.7]        30.8 [30.7-30.9]        50.0x         1.3x         1.4x
```

## rms_norm (float16)

candidate: hipbridge.kernels.norm.rms_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       202.8 [202.7-202.8]           2.8 [2.8-2.8]           7.4 [7.1-7.8]        16.2 [16.2-16.3]        12.5x         0.2x         0.5x  (latency-bound)
   1024x1024       228.3 [228.2-228.4]           4.2 [4.1-4.2]           7.8 [7.5-7.9]        16.5 [16.3-16.7]        13.8x         0.3x         0.5x  (latency-bound)
   4096x4096    1055.0 [1054.8-1055.1]        22.9 [22.9-22.9]        19.7 [19.6-20.1]        16.8 [16.8-17.0]        62.8x         1.4x         1.2x
```

## rms_norm (bfloat16)

candidate: hipbridge.kernels.norm.rms_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       220.2 [220.2-220.4]           2.8 [2.8-2.8]           7.2 [7.1-7.4]        16.5 [16.2-16.6]        13.3x         0.2x         0.4x  (latency-bound)
   1024x1024       250.6 [250.5-250.7]           4.2 [4.2-4.2]           7.4 [7.4-7.9]        16.6 [16.6-16.7]        15.1x         0.3x         0.4x  (latency-bound)
   4096x4096    1145.8 [1145.8-1146.3]        24.7 [24.7-24.8]        21.2 [20.5-21.6]        17.7 [17.6-17.7]        64.9x         1.4x         1.2x
```

## rms_norm_affine (float32)

candidate: hipbridge.kernels.norm.rms_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       173.6 [173.6-173.8]           2.8 [2.8-2.9]           7.7 [7.4-7.9]        18.0 [17.6-18.1]         9.6x         0.2x         0.4x  (latency-bound)
   1024x1024       200.4 [200.3-200.7]           4.0 [4.0-4.0]           7.7 [7.7-7.8]        17.8 [17.6-18.5]        11.2x         0.2x         0.4x  (latency-bound)
   4096x4096    1669.2 [1658.8-1669.9]        41.5 [41.3-41.7]        43.9 [43.7-44.1]        31.8 [31.1-31.9]        52.4x         1.3x         1.4x
```

## rms_norm_affine (float16)

candidate: hipbridge.kernels.norm.rms_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       214.7 [214.6-214.7]           2.9 [2.8-2.9]           7.1 [7.1-7.3]        17.9 [17.8-17.9]        12.0x         0.2x         0.4x  (latency-bound)
   1024x1024       240.7 [240.2-240.7]           4.2 [4.2-4.2]           7.7 [7.7-8.3]        18.3 [18.1-18.4]        13.2x         0.2x         0.4x  (latency-bound)
   4096x4096    1128.4 [1127.8-1128.4]        24.6 [24.5-24.9]        21.3 [20.7-22.1]        18.0 [18.0-18.1]        62.6x         1.4x         1.2x
```

## rms_norm_affine (bfloat16)

candidate: hipbridge.kernels.norm.rms_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       233.9 [233.9-234.0]           2.9 [2.9-2.9]           8.2 [8.1-8.6]        19.0 [18.0-20.0]        12.3x         0.2x         0.4x  (latency-bound)
   1024x1024       266.6 [265.8-266.9]           4.3 [4.3-4.3]           7.8 [7.6-8.2]        18.5 [18.3-18.6]        14.4x         0.2x         0.4x  (latency-bound)
   4096x4096    1245.3 [1244.9-1245.3]        26.4 [26.2-26.7]        23.0 [21.7-23.4]        18.2 [18.0-18.3]        68.5x         1.5x         1.3x
```

## rope (float32)

candidate: hipbridge.kernels.rope (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024          80.5 [80.5-80.6]           2.2 [2.2-2.3]        49.5 [47.5-49.9]        20.1 [19.8-20.2]         4.0x         0.1x         2.5x  (latency-bound)
   1024x1024       102.0 [101.8-102.1]           2.8 [2.8-2.8]        50.7 [47.4-52.0]        20.2 [19.9-21.0]         5.1x         0.1x         2.5x  (latency-bound)
   4096x4096    1373.8 [1373.5-1374.5]        45.8 [45.3-45.9]     241.7 [240.8-243.1]        52.9 [52.8-52.9]        26.0x         0.9x         4.6x
```

## rope (float16)

candidate: hipbridge.kernels.rope (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024          79.5 [79.4-79.5]           2.2 [2.2-2.2]        48.5 [47.1-49.6]        19.9 [19.9-20.1]         4.0x         0.1x         2.4x  (latency-bound)
   1024x1024          97.0 [97.0-97.1]           2.7 [2.7-2.7]        47.6 [46.5-48.1]        19.7 [19.6-19.7]         4.9x         0.1x         2.4x  (latency-bound)
   4096x4096       734.5 [732.4-735.2]        23.8 [23.6-24.2]     139.1 [138.9-141.8]        26.7 [26.7-26.8]        27.5x         0.9x         5.2x
```

## rope (bfloat16)

candidate: hipbridge.kernels.rope (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       109.5 [109.5-109.6]           2.3 [2.2-2.3]        47.5 [47.0-56.3]        24.4 [24.1-24.5]         4.5x         0.1x         1.9x  (latency-bound)
   1024x1024       127.3 [127.3-127.3]           2.8 [2.8-2.8]        49.9 [48.8-50.3]        20.1 [20.1-20.2]         6.3x         0.1x         2.5x  (latency-bound)
   4096x4096       934.1 [933.5-947.8]        24.1 [23.8-24.1]     139.9 [138.9-142.0]        26.9 [26.8-27.1]        34.7x         0.9x         5.2x
```

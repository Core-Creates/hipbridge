# hipbridge benchmark

| field | value |
|---|---|
| generated | 2026-09-08 04:34 UTC |
| hipbridge | 0.1.0.dev0 at commit `13acca4` |
| host | 2 (Linux x86_64) |
| device | AMD Radeon Graphics |
| toolchain | hipcc, HIP version: 7.14.60850-0000000 |
| arch | gfx942 |
| torch | 2.9.1+rocm6.4 |
| measured code | `4736047d7ba708bf` |

## row_softmax (float32)

candidate: hipbridge.kernels.softmax (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       325.5 [325.4-325.6]           4.1 [4.1-4.1]           5.2 [5.0-5.3]        16.1 [16.1-16.1]        20.3x         0.3x         0.3x  (latency-bound)
   1024x1024       364.1 [363.9-364.2]           6.3 [6.3-6.4]           5.2 [5.2-5.4]        16.3 [16.1-16.3]        22.3x         0.4x         0.3x  (latency-bound)
   4096x4096    2433.3 [2433.0-2434.1]        53.1 [52.5-53.4]        39.2 [38.8-39.8]        31.2 [31.2-31.2]        77.9x         1.7x         1.3x
```

## row_softmax (float16)

candidate: hipbridge.kernels.softmax (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       309.0 [308.9-309.2]           4.0 [4.0-4.0]           5.2 [5.2-5.3]        16.2 [15.9-16.3]        19.1x         0.2x         0.3x  (latency-bound)
   1024x1024       341.8 [341.8-342.1]           6.4 [6.4-6.4]           6.7 [6.7-6.7]        16.1 [16.1-16.2]        21.2x         0.4x         0.4x  (latency-bound)
   4096x4096    2744.2 [2743.4-2744.3]        45.2 [44.8-45.3]        24.3 [23.5-25.5]        24.1 [23.6-24.3]       113.9x         1.9x         1.0x
```

## row_softmax (bfloat16)

candidate: hipbridge.kernels.softmax (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       339.9 [339.8-340.0]           4.1 [4.1-4.1]           5.6 [5.6-5.6]        16.2 [16.0-16.4]        21.0x         0.3x         0.3x  (latency-bound)
   1024x1024       372.3 [372.1-372.3]           6.6 [6.5-6.6]           7.2 [7.2-7.2]        16.0 [15.9-16.3]        23.3x         0.4x         0.4x  (latency-bound)
   4096x4096    3223.5 [3223.4-3224.0]        46.3 [45.4-46.6]        26.1 [25.3-28.3]        25.2 [24.8-25.8]       127.7x         1.8x         1.0x
```

## layer_norm (float32)

candidate: hipbridge.kernels.norm.layer_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       204.9 [204.8-204.9]           4.0 [3.9-4.1]           7.9 [7.9-8.0]        16.8 [16.4-16.9]        12.2x         0.2x         0.5x  (latency-bound)
   1024x1024       233.5 [233.4-233.6]           5.4 [5.4-5.4]           8.2 [8.0-8.2]        16.5 [16.4-16.8]        14.2x         0.3x         0.5x  (latency-bound)
   4096x4096    1746.6 [1730.0-1747.9]        46.2 [46.2-46.3]        44.9 [44.8-45.0]        30.7 [30.5-30.9]        57.0x         1.5x         1.5x
```

## layer_norm (float16)

candidate: hipbridge.kernels.norm.layer_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       272.0 [271.9-272.2]           3.9 [3.9-4.0]           8.2 [8.1-8.3]        16.3 [16.3-16.4]        16.7x         0.2x         0.5x  (latency-bound)
   1024x1024       297.3 [297.2-297.6]           5.6 [5.6-5.6]           8.4 [8.2-8.5]        16.5 [16.5-16.7]        18.0x         0.3x         0.5x  (latency-bound)
   4096x4096    1368.5 [1367.3-1368.9]        31.0 [30.7-31.1]        28.0 [27.2-29.8]        17.7 [17.5-17.8]        77.3x         1.8x         1.6x
```

## layer_norm (bfloat16)

candidate: hipbridge.kernels.norm.layer_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       291.6 [291.2-291.6]           3.9 [3.9-4.1]           8.2 [8.1-8.5]        16.4 [16.2-16.5]        17.8x         0.2x         0.5x  (latency-bound)
   1024x1024       317.4 [317.3-317.4]           5.6 [5.6-5.6]           8.2 [8.2-8.5]        16.7 [16.5-17.0]        19.0x         0.3x         0.5x  (latency-bound)
   4096x4096    1486.5 [1485.6-1486.9]        31.7 [31.4-32.0]        29.1 [28.6-30.7]        18.5 [18.5-18.8]        80.3x         1.7x         1.6x
```

## layer_norm_affine (float32)

candidate: hipbridge.kernels.norm.layer_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       209.0 [209.0-209.1]           3.9 [3.9-3.9]         10.1 [9.8-10.1]        22.1 [21.6-22.1]         9.5x         0.2x         0.5x  (latency-bound)
   1024x1024       241.0 [241.0-241.1]           5.5 [5.5-5.5]           8.3 [8.2-8.4]        19.5 [19.3-20.2]        12.4x         0.3x         0.4x  (latency-bound)
   4096x4096    1856.7 [1831.3-1859.7]        46.8 [46.8-47.2]        45.5 [45.0-46.5]        32.6 [32.6-33.1]        56.9x         1.4x         1.4x
```

## layer_norm_affine (float16)

candidate: hipbridge.kernels.norm.layer_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       276.0 [275.9-276.2]           4.0 [4.0-4.1]           8.3 [8.3-8.3]        19.2 [19.0-19.4]        14.4x         0.2x         0.4x  (latency-bound)
   1024x1024       306.0 [305.6-306.3]           5.7 [5.7-5.7]           8.5 [8.4-8.8]        19.4 [19.2-19.5]        15.8x         0.3x         0.4x  (latency-bound)
   4096x4096    1425.2 [1424.8-1425.9]        33.3 [33.2-33.7]        30.3 [29.7-32.1]        24.3 [24.3-24.4]        58.5x         1.4x         1.2x
```

## layer_norm_affine (bfloat16)

candidate: hipbridge.kernels.norm.layer_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       301.2 [300.9-301.3]           4.0 [3.9-4.0]           8.4 [8.4-8.7]        19.6 [19.6-19.8]        15.3x         0.2x         0.4x  (latency-bound)
   1024x1024       332.6 [331.7-332.9]           5.8 [5.7-5.9]           8.3 [8.2-8.6]        19.7 [19.3-19.7]        16.9x         0.3x         0.4x  (latency-bound)
   4096x4096    1580.4 [1580.3-1581.7]        35.3 [35.3-35.4]        31.7 [30.9-32.6]        20.5 [20.4-20.9]        77.0x         1.7x         1.5x
```

## rms_norm (float32)

candidate: hipbridge.kernels.norm.rms_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       169.7 [169.5-170.0]           2.8 [2.7-2.8]           6.8 [6.8-7.0]        16.6 [16.6-16.7]        10.2x         0.2x         0.4x  (latency-bound)
   1024x1024       193.6 [193.6-193.7]           4.0 [4.0-4.0]           7.2 [7.2-7.2]        16.9 [16.9-16.9]        11.5x         0.2x         0.4x  (latency-bound)
   4096x4096    1446.0 [1445.9-1447.5]        41.4 [41.1-41.4]        42.6 [42.6-42.7]        31.1 [30.9-31.2]        46.5x         1.3x         1.4x
```

## rms_norm (float16)

candidate: hipbridge.kernels.norm.rms_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       202.8 [202.7-202.9]           2.7 [2.7-2.8]           6.9 [6.8-7.0]        16.4 [16.3-16.6]        12.4x         0.2x         0.4x  (latency-bound)
   1024x1024       227.1 [227.1-227.4]           4.1 [4.1-4.1]           8.0 [8.0-8.5]        18.9 [18.6-19.0]        12.0x         0.2x         0.4x  (latency-bound)
   4096x4096    1054.8 [1053.6-1055.0]        22.8 [22.6-22.8]        19.9 [19.7-20.1]        20.5 [20.3-20.7]        51.5x         1.1x         1.0x
```

## rms_norm (bfloat16)

candidate: hipbridge.kernels.norm.rms_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       220.2 [220.2-220.3]           2.8 [2.7-2.8]           7.0 [6.9-7.0]        16.7 [16.5-16.8]        13.2x         0.2x         0.4x  (latency-bound)
   1024x1024       248.2 [247.7-250.6]           4.2 [4.2-4.3]           7.3 [7.2-8.3]        17.0 [16.9-18.6]        14.6x         0.2x         0.4x  (latency-bound)
   4096x4096    1142.3 [1141.9-1143.3]        24.7 [24.5-24.7]        21.5 [20.4-21.8]        17.0 [17.0-17.3]        67.1x         1.4x         1.3x
```

## rms_norm_affine (float32)

candidate: hipbridge.kernels.norm.rms_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       173.6 [173.4-173.7]           2.9 [2.9-3.0]           7.0 [6.9-7.1]        17.8 [17.8-18.2]         9.8x         0.2x         0.4x  (latency-bound)
   1024x1024       200.7 [200.7-200.7]           4.0 [4.0-4.1]           7.7 [7.7-7.8]        18.2 [18.2-18.6]        11.0x         0.2x         0.4x  (latency-bound)
   4096x4096    1574.3 [1569.0-1595.6]        41.5 [41.4-41.7]        43.2 [43.1-43.6]        31.7 [31.6-31.7]        49.7x         1.3x         1.4x
```

## rms_norm_affine (float16)

candidate: hipbridge.kernels.norm.rms_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       214.6 [214.6-214.6]           3.0 [3.0-3.1]           6.8 [6.8-6.9]        17.9 [17.7-17.9]        12.0x         0.2x         0.4x  (latency-bound)
   1024x1024       240.8 [239.6-241.0]           4.2 [4.2-4.2]           7.2 [7.1-7.4]        18.0 [17.9-18.2]        13.4x         0.2x         0.4x  (latency-bound)
   4096x4096    1127.1 [1126.9-1127.6]        25.0 [24.7-25.1]        21.8 [20.7-21.8]        22.6 [21.8-24.3]        49.9x         1.1x         1.0x
```

## rms_norm_affine (bfloat16)

candidate: hipbridge.kernels.norm.rms_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       234.2 [234.0-234.2]           2.9 [2.9-3.1]           7.1 [7.1-7.2]        18.0 [17.8-18.2]        13.0x         0.2x         0.4x  (latency-bound)
   1024x1024       269.0 [268.5-269.1]           4.3 [4.3-4.3]           7.2 [7.1-7.4]        18.2 [18.2-18.3]        14.8x         0.2x         0.4x  (latency-bound)
   4096x4096    1250.5 [1249.5-1250.6]        26.2 [26.1-26.5]        22.7 [21.7-22.7]        18.0 [17.9-18.2]        69.3x         1.5x         1.3x
```

## rope (float32)

candidate: hipbridge.kernels.rope (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024          80.4 [80.3-80.5]           2.2 [2.2-2.3]        47.3 [45.9-48.3]        19.5 [19.4-19.7]         4.1x         0.1x         2.4x  (latency-bound)
   1024x1024       101.9 [101.8-102.1]           2.8 [2.8-2.8]        45.9 [45.5-46.9]        19.8 [19.7-21.5]         5.2x         0.1x         2.3x  (latency-bound)
   4096x4096    1372.7 [1372.6-1373.1]        45.8 [45.6-46.3]     242.5 [241.7-243.3]        53.0 [53.0-53.2]        25.9x         0.9x         4.6x
```

## rope (float16)

candidate: hipbridge.kernels.rope (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024          79.5 [79.5-79.5]           2.2 [2.2-2.2]        47.7 [46.7-48.1]        23.4 [23.1-24.7]         3.4x         0.1x         2.0x  (latency-bound)
   1024x1024          97.1 [97.0-97.2]           2.7 [2.7-2.8]        45.9 [45.4-46.4]        19.7 [19.6-19.7]         4.9x         0.1x         2.3x  (latency-bound)
   4096x4096       743.4 [743.4-870.7]        23.7 [23.6-23.9]     138.9 [138.4-142.0]        27.5 [27.0-27.5]        27.0x         0.9x         5.0x
```

## rope (bfloat16)

candidate: hipbridge.kernels.rope (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       109.5 [109.5-109.6]           2.2 [2.2-2.2]        46.9 [45.9-48.0]        19.6 [19.3-19.7]         5.6x         0.1x         2.4x  (latency-bound)
   1024x1024       127.3 [127.3-127.4]           2.8 [2.8-2.9]        47.0 [46.8-48.4]        19.7 [19.7-20.0]         6.5x         0.1x         2.4x  (latency-bound)
   4096x4096       934.6 [931.5-935.3]        23.4 [23.4-23.6]     139.1 [138.6-141.4]        27.4 [27.3-27.8]        34.2x         0.9x         5.1x
```

## rms_norm_rope (float32)

candidate: hipbridge.kernels.fused.rms_norm_rope (Triton, fused)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)         2 launches (us)          candidate (us)  vs original vs tuned HIPvs 2 launches
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       120.6 [120.6-120.7]           2.7 [2.7-2.8]        40.1 [40.1-40.2]        22.2 [22.1-22.3]         5.4x         0.1x         1.8x  (latency-bound)
   1024x1024       140.8 [140.2-140.8]           4.3 [4.3-4.3]        40.3 [40.0-40.4]        22.4 [22.3-22.7]         6.3x         0.2x         1.8x  (latency-bound)
   4096x4096    2196.5 [2192.3-2198.6]        58.7 [58.2-58.7]        85.4 [85.4-87.5]        72.4 [71.8-73.1]        30.3x         0.8x         1.2x
```

## rms_norm_rope (float16)

candidate: hipbridge.kernels.fused.rms_norm_rope (Triton, fused)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)         2 launches (us)          candidate (us)  vs original vs tuned HIPvs 2 launches
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       151.7 [151.6-151.7]           2.7 [2.6-2.7]        40.1 [40.0-40.3]        21.9 [21.8-22.0]         6.9x         0.1x         1.8x  (latency-bound)
   1024x1024       172.6 [170.1-172.9]           4.1 [4.1-4.1]        40.0 [40.0-40.2]        21.9 [21.9-22.2]         7.9x         0.2x         1.8x  (latency-bound)
   4096x4096    1164.0 [1160.9-1164.9]        29.5 [29.4-29.9]        47.6 [47.6-48.2]        41.0 [40.6-42.4]        28.4x         0.7x         1.2x
```

## rms_norm_rope (bfloat16)

candidate: hipbridge.kernels.fused.rms_norm_rope (Triton, fused)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)         2 launches (us)          candidate (us)  vs original vs tuned HIPvs 2 launches
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       183.7 [183.6-183.7]           2.7 [2.6-2.7]        41.3 [41.2-41.7]        22.8 [22.5-22.9]         8.1x         0.1x         1.8x  (latency-bound)
   1024x1024       202.7 [202.6-202.7]           4.2 [4.2-4.2]        41.2 [41.2-41.3]        22.9 [22.3-23.0]         8.9x         0.2x         1.8x  (latency-bound)
   4096x4096    1358.5 [1357.9-1360.5]        30.0 [29.8-30.2]        48.1 [47.0-49.1]        41.8 [41.2-42.6]        32.5x         0.7x         1.1x
```

# hipbridge benchmark

| field | value |
|---|---|
| generated | 2026-09-08 20:34 UTC |
| hipbridge | 0.1.0.dev0 at commit `b333833` |
| host | 2 (Linux x86_64) |
| device | AMD Radeon Graphics |
| toolchain | hipcc, HIP version: 7.14.60850-0000000 |
| arch | gfx942 |
| torch | 2.9.1+rocm6.4 |
| measured code | `2ecfdefc69200090` |

## row_softmax (float32)

candidate: hipbridge.kernels.softmax (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       322.3 [322.1-322.4]           4.0 [4.0-4.1]           5.2 [5.1-5.3]        17.7 [17.5-17.7]        18.2x         0.2x         0.3x  (latency-bound)
   1024x1024       388.3 [387.8-388.8]           6.3 [6.3-6.3]           5.3 [5.2-5.3]        17.6 [17.5-17.7]        22.1x         0.4x         0.3x  (latency-bound)
   4096x4096    2433.7 [2433.3-2433.9]        53.3 [52.6-53.5]        40.4 [40.0-41.3]        31.8 [31.5-31.8]        76.5x         1.7x         1.3x
```

## row_softmax (float16)

candidate: hipbridge.kernels.softmax (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       308.0 [307.9-308.1]           3.9 [3.9-4.0]           5.3 [5.2-5.5]        18.5 [18.1-23.6]        16.6x         0.2x         0.3x  (latency-bound)
   1024x1024       349.6 [349.4-358.5]           6.4 [6.4-6.5]           6.7 [6.7-6.8]        18.0 [17.3-18.7]        19.4x         0.4x         0.4x  (latency-bound)
   4096x4096    2737.8 [2737.7-2738.6]        45.4 [45.0-45.7]        24.9 [23.6-25.0]        24.3 [24.1-24.6]       112.8x         1.9x         1.0x
```

## row_softmax (bfloat16)

candidate: hipbridge.kernels.softmax (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       339.5 [339.4-339.6]           4.1 [4.1-4.1]           5.8 [5.6-5.8]        17.9 [17.8-18.2]        19.0x         0.2x         0.3x  (latency-bound)
   1024x1024       372.4 [372.3-372.7]           6.5 [6.5-6.6]           7.2 [7.2-7.2]        16.0 [15.9-16.3]        23.2x         0.4x         0.4x  (latency-bound)
   4096x4096    3219.5 [3218.9-3219.8]        46.0 [45.7-46.3]        25.9 [25.2-26.7]        25.5 [25.3-25.6]       126.4x         1.8x         1.0x
```

## layer_norm (float32)

candidate: hipbridge.kernels.norm.layer_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       204.9 [204.9-205.0]           4.0 [3.9-4.0]           8.6 [8.4-8.7]        17.8 [17.7-18.1]        11.5x         0.2x         0.5x  (latency-bound)
   1024x1024       233.5 [233.4-233.5]           5.4 [5.4-5.4]           9.7 [9.4-9.7]        18.6 [18.4-18.7]        12.6x         0.3x         0.5x  (latency-bound)
   4096x4096    1742.1 [1728.6-1750.8]        45.7 [45.6-46.0]        44.7 [44.6-45.0]        31.0 [31.0-31.0]        56.2x         1.5x         1.4x
```

## layer_norm (float16)

candidate: hipbridge.kernels.norm.layer_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       271.7 [271.7-271.9]           4.0 [4.0-4.1]           9.2 [9.2-9.3]        18.3 [17.8-18.5]        14.8x         0.2x         0.5x  (latency-bound)
   1024x1024       297.6 [297.0-297.8]           5.6 [5.6-5.6]           8.8 [8.5-9.9]        17.6 [16.7-18.3]        16.9x         0.3x         0.5x  (latency-bound)
   4096x4096    1370.1 [1370.0-1371.2]        30.8 [30.3-31.1]        28.1 [27.3-29.6]        17.5 [17.5-17.5]        78.2x         1.8x         1.6x
```

## layer_norm (bfloat16)

candidate: hipbridge.kernels.norm.layer_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       291.5 [291.2-291.5]           3.9 [3.9-4.0]           9.1 [9.0-9.9]        20.3 [20.2-20.3]        14.4x         0.2x         0.5x  (latency-bound)
   1024x1024       317.9 [317.7-318.7]           5.6 [5.6-5.7]           8.9 [8.9-9.2]        18.0 [17.8-18.1]        17.7x         0.3x         0.5x  (latency-bound)
   4096x4096    1487.4 [1486.6-1488.6]        31.8 [31.7-32.3]        28.7 [28.6-30.7]        18.8 [18.4-18.9]        79.1x         1.7x         1.5x
```

## layer_norm_affine (float32)

candidate: hipbridge.kernels.norm.layer_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       208.9 [208.6-209.1]           4.0 [3.9-4.1]           9.1 [9.1-9.1]        21.4 [20.9-21.6]         9.8x         0.2x         0.4x  (latency-bound)
   1024x1024       240.9 [240.7-241.0]           5.5 [5.5-5.5]           9.1 [9.0-9.3]        21.7 [21.5-21.8]        11.1x         0.3x         0.4x  (latency-bound)
   4096x4096    1844.9 [1837.0-1851.6]        46.8 [46.7-47.3]        45.3 [45.0-45.7]        33.2 [33.2-33.2]        55.6x         1.4x         1.4x
```

## layer_norm_affine (float16)

candidate: hipbridge.kernels.norm.layer_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       276.1 [275.9-276.1]           4.0 [4.0-4.1]        10.1 [10.1-10.4]        24.3 [24.3-25.2]        11.3x         0.2x         0.4x  (latency-bound)
   1024x1024       306.3 [305.9-306.5]           5.7 [5.7-5.7]        10.3 [10.1-10.3]        24.4 [24.0-24.4]        12.6x         0.2x         0.4x  (latency-bound)
   4096x4096    1422.2 [1422.1-1422.9]        33.6 [33.6-34.0]        29.9 [29.7-31.7]        22.3 [22.1-23.6]        63.7x         1.5x         1.3x
```

## layer_norm_affine (bfloat16)

candidate: hipbridge.kernels.norm.layer_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       301.1 [301.0-301.1]           4.0 [4.0-4.0]           8.9 [8.8-8.9]        21.3 [21.3-21.8]        14.1x         0.2x         0.4x  (latency-bound)
   1024x1024       332.4 [331.9-333.0]           5.7 [5.7-5.8]           8.4 [8.2-8.5]        19.3 [19.2-19.4]        17.2x         0.3x         0.4x  (latency-bound)
   4096x4096    1580.2 [1579.4-1580.9]        35.3 [34.8-35.4]        31.4 [31.0-32.7]        20.7 [20.5-20.8]        76.4x         1.7x         1.5x
```

## rms_norm (float32)

candidate: hipbridge.kernels.norm.rms_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       169.8 [169.7-169.9]           2.8 [2.8-2.8]           7.8 [7.7-8.0]        18.4 [18.4-18.4]         9.2x         0.2x         0.4x  (latency-bound)
   1024x1024       193.7 [193.5-193.7]           4.1 [4.0-4.1]           7.7 [7.6-8.0]        18.5 [18.4-19.3]        10.5x         0.2x         0.4x  (latency-bound)
   4096x4096    1459.7 [1454.7-1462.4]        41.3 [41.1-41.4]        42.8 [42.7-42.8]        31.1 [31.1-31.1]        47.0x         1.3x         1.4x
```

## rms_norm (float16)

candidate: hipbridge.kernels.norm.rms_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       203.0 [202.8-203.1]           2.8 [2.7-2.8]           7.6 [7.6-7.7]        16.3 [16.1-16.8]        12.5x         0.2x         0.5x  (latency-bound)
   1024x1024       227.7 [227.6-227.8]           4.1 [4.1-4.1]           8.8 [8.4-9.0]        20.3 [20.2-20.6]        11.2x         0.2x         0.4x  (latency-bound)
   4096x4096    1055.6 [1055.1-1056.6]        22.6 [22.6-22.8]        20.1 [19.7-20.3]        17.0 [16.9-17.1]        62.2x         1.3x         1.2x
```

## rms_norm (bfloat16)

candidate: hipbridge.kernels.norm.rms_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       220.0 [219.9-220.1]           2.9 [2.8-2.9]           6.9 [6.9-7.0]        16.6 [16.5-16.7]        13.3x         0.2x         0.4x  (latency-bound)
   1024x1024       245.7 [245.6-248.4]           4.2 [4.2-4.3]           7.4 [7.2-7.4]        16.6 [16.4-16.6]        14.8x         0.3x         0.4x  (latency-bound)
   4096x4096    1139.1 [1138.2-1140.3]        24.6 [24.5-24.9]        21.9 [20.4-21.9]        17.7 [17.6-17.9]        64.2x         1.4x         1.2x
```

## rms_norm_affine (float32)

candidate: hipbridge.kernels.norm.rms_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       173.9 [173.8-173.9]           2.9 [2.9-2.9]           7.0 [6.9-7.0]        18.3 [18.2-19.7]         9.5x         0.2x         0.4x  (latency-bound)
   1024x1024       200.5 [200.3-200.6]           4.0 [4.0-4.0]           7.6 [7.6-7.7]        18.3 [18.1-20.1]        10.9x         0.2x         0.4x  (latency-bound)
   4096x4096    1571.5 [1569.4-1575.5]        41.6 [41.5-41.6]        43.5 [43.2-43.6]        32.0 [31.5-32.1]        49.2x         1.3x         1.4x
```

## rms_norm_affine (float16)

candidate: hipbridge.kernels.norm.rms_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       214.6 [214.5-214.7]           2.8 [2.8-2.8]           7.1 [7.1-7.1]        17.9 [17.8-18.1]        12.0x         0.2x         0.4x  (latency-bound)
   1024x1024       239.3 [239.1-239.5]           4.2 [4.2-4.2]           7.3 [7.2-7.6]        18.2 [18.1-18.3]        13.1x         0.2x         0.4x  (latency-bound)
   4096x4096    1127.2 [1126.4-1128.3]        24.9 [24.8-25.2]        21.5 [20.8-21.6]        18.2 [18.1-18.3]        61.8x         1.4x         1.2x
```

## rms_norm_affine (bfloat16)

candidate: hipbridge.kernels.norm.rms_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       234.0 [234.0-234.1]           2.9 [2.9-2.9]           7.2 [7.1-7.2]        18.5 [18.5-18.6]        12.7x         0.2x         0.4x  (latency-bound)
   1024x1024       264.4 [263.1-264.7]           4.2 [4.2-4.3]           7.2 [7.2-7.4]        18.4 [18.3-18.6]        14.4x         0.2x         0.4x  (latency-bound)
   4096x4096    1248.0 [1247.5-1248.5]        26.7 [26.3-26.9]        22.8 [21.8-23.1]        18.5 [18.3-20.7]        67.4x         1.4x         1.2x
```

## rope (float32)

candidate: hipbridge.kernels.rope (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024          80.4 [80.4-80.4]           2.3 [2.3-2.5]        50.5 [48.3-51.4]        21.7 [21.7-22.8]         3.7x         0.1x         2.3x  (latency-bound)
   1024x1024       102.0 [101.9-102.0]           2.8 [2.8-2.8]        46.8 [46.1-47.5]        20.2 [20.1-20.6]         5.1x         0.1x         2.3x  (latency-bound)
   4096x4096    1373.2 [1372.4-1373.4]        46.1 [45.9-46.1]     237.5 [236.8-238.7]        50.4 [50.4-50.8]        27.2x         0.9x         4.7x
```

## rope (float16)

candidate: hipbridge.kernels.rope (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024          79.4 [79.3-79.4]           2.3 [2.2-2.4]        49.8 [46.6-52.6]        22.0 [21.9-22.3]         3.6x         0.1x         2.3x  (latency-bound)
   1024x1024          97.1 [97.1-97.1]           2.7 [2.7-2.7]        48.0 [47.8-48.9]        20.1 [19.9-20.2]         4.8x         0.1x         2.4x  (latency-bound)
   4096x4096       723.3 [723.1-727.8]        23.6 [23.6-23.8]     142.7 [142.2-147.7]        27.9 [27.5-27.9]        26.0x         0.8x         5.1x
```

## rope (bfloat16)

candidate: hipbridge.kernels.rope (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       109.6 [109.5-109.6]           2.3 [2.2-2.4]        48.9 [47.9-52.7]        20.0 [20.0-20.2]         5.5x         0.1x         2.4x  (latency-bound)
   1024x1024       127.4 [127.4-127.4]           2.8 [2.8-2.8]        48.4 [46.4-49.2]        22.9 [22.2-23.6]         5.6x         0.1x         2.1x  (latency-bound)
   4096x4096       930.4 [930.4-933.8]        23.6 [23.4-23.7]     141.8 [141.8-145.0]        28.3 [28.0-28.4]        32.8x         0.8x         5.0x
```

## rms_norm_rope (float32)

candidate: hipbridge.kernels.fused.rms_norm_rope (Triton, fused)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)         2 launches (us)          candidate (us)  vs original vs tuned HIPvs 2 launches
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       120.5 [120.4-120.6]           2.7 [2.7-2.9]       40.1 [39.6-111.1]        24.5 [24.4-24.7]         4.9x         0.1x         1.6x  (latency-bound)
   1024x1024       140.7 [140.4-140.7]           4.3 [4.3-4.3]        39.7 [39.6-40.4]        22.1 [22.0-22.2]         6.4x         0.2x         1.8x  (latency-bound)
   4096x4096    2176.4 [2172.2-2185.6]        58.6 [58.4-58.9]        86.0 [85.3-87.0]        72.2 [72.2-73.2]        30.1x         0.8x         1.2x
```

## rms_norm_rope (float16)

candidate: hipbridge.kernels.fused.rms_norm_rope (Triton, fused)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)         2 launches (us)          candidate (us)  vs original vs tuned HIPvs 2 launches
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       151.6 [151.5-169.1]           2.8 [2.7-2.8]        40.2 [39.9-43.7]        22.2 [22.0-22.3]         6.8x         0.1x         1.8x  (latency-bound)
   1024x1024       170.1 [169.9-172.7]           4.1 [4.1-4.1]        40.2 [39.7-41.2]        22.3 [22.2-22.4]         7.6x         0.2x         1.8x  (latency-bound)
   4096x4096    1159.6 [1158.3-1232.4]        29.7 [29.6-29.8]        48.0 [47.1-48.1]        41.2 [40.7-42.4]        28.1x         0.7x         1.2x
```

## rms_norm_rope (bfloat16)

candidate: hipbridge.kernels.fused.rms_norm_rope (Triton, fused)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)         2 launches (us)          candidate (us)  vs original vs tuned HIPvs 2 launches
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       183.8 [183.8-183.8]           2.7 [2.7-2.8]        44.1 [41.2-45.5]        22.2 [22.1-22.4]         8.3x         0.1x         2.0x  (latency-bound)
   1024x1024       203.3 [203.3-203.3]           4.2 [4.2-4.2]        47.6 [44.3-49.1]        27.2 [26.7-27.4]         7.5x         0.2x         1.8x  (latency-bound)
   4096x4096    1342.3 [1192.5-1346.5]        30.3 [29.8-30.4]        48.6 [47.9-50.0]        41.3 [41.0-42.9]        32.5x         0.7x         1.2x
```

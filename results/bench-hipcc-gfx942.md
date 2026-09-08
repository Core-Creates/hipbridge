# hipbridge benchmark

| field | value |
|---|---|
| generated | 2026-09-08 22:55 UTC |
| hipbridge | 0.1.0.dev0 at commit `8c1834e` |
| host | 2 (Linux x86_64) |
| device | AMD Radeon Graphics |
| toolchain | hipcc, HIP version: 7.14.60850-0000000 |
| arch | gfx942 |
| torch | 2.9.1+rocm6.4 |
| measured code | `7aa3e84fc2a962b5` |

## row_softmax (float32)

candidate: hipbridge.kernels.softmax (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       321.9 [321.7-322.3]           4.0 [4.0-4.0]           5.3 [5.2-5.6]        18.7 [18.7-20.1]        17.2x         0.2x         0.3x  (latency-bound)
   1024x1024       363.9 [363.7-364.0]           6.3 [6.3-6.4]           5.8 [5.6-6.1]        20.7 [20.5-21.0]        17.6x         0.3x         0.3x  (latency-bound)
   4096x4096    2427.2 [2426.5-2427.9]        53.1 [52.0-53.5]        39.2 [38.8-39.8]        32.6 [32.6-32.7]        74.4x         1.6x         1.2x
```

## row_softmax (float16)

candidate: hipbridge.kernels.softmax (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       309.1 [308.9-309.1]           4.0 [4.0-4.0]           6.0 [5.7-6.0]        20.2 [19.7-20.6]        15.3x         0.2x         0.3x  (latency-bound)
   1024x1024       341.7 [341.7-341.8]           6.4 [6.4-6.4]           6.7 [6.7-6.8]        20.4 [20.0-20.6]        16.7x         0.3x         0.3x  (latency-bound)
   4096x4096    2743.5 [2741.7-2744.3]        45.7 [44.6-45.7]        25.3 [23.5-25.3]        23.9 [23.6-24.0]       115.0x         1.9x         1.1x
```

## row_softmax (bfloat16)

candidate: hipbridge.kernels.softmax (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       339.5 [339.3-339.6]           4.1 [4.0-4.1]           5.6 [5.5-5.7]        18.3 [18.3-18.5]        18.5x         0.2x         0.3x  (latency-bound)
   1024x1024       372.3 [372.3-372.7]           6.6 [6.5-6.6]           7.2 [7.1-7.2]        18.5 [18.4-18.6]        20.1x         0.4x         0.4x  (latency-bound)
   4096x4096    3223.4 [3223.2-3225.0]        45.9 [45.1-46.1]        26.2 [25.2-27.1]        25.2 [25.0-25.8]       128.1x         1.8x         1.0x
```

## layer_norm (float32)

candidate: hipbridge.kernels.norm.layer_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       205.0 [204.7-205.0]           4.0 [3.9-4.1]           8.1 [8.1-8.1]        18.7 [18.5-20.5]        10.9x         0.2x         0.4x  (latency-bound)
   1024x1024       233.4 [233.4-233.6]           5.4 [5.4-5.4]           8.2 [8.1-8.4]        20.3 [18.6-22.8]        11.5x         0.3x         0.4x  (latency-bound)
   4096x4096    1746.6 [1743.8-1747.7]        45.9 [45.7-46.1]        44.8 [44.5-45.2]        31.3 [31.0-31.3]        55.8x         1.5x         1.4x
```

## layer_norm (float16)

candidate: hipbridge.kernels.norm.layer_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       271.9 [271.8-272.0]           4.0 [4.0-4.1]          9.6 [9.5-10.2]        20.7 [18.6-20.8]        13.1x         0.2x         0.5x  (latency-bound)
   1024x1024       297.2 [297.2-297.3]           5.6 [5.5-5.6]           9.5 [9.1-9.8]        19.3 [19.1-19.4]        15.4x         0.3x         0.5x  (latency-bound)
   4096x4096    1369.9 [1369.4-1370.3]        30.7 [30.4-31.0]        27.6 [27.5-29.2]        23.2 [23.0-23.3]        59.0x         1.3x         1.2x
```

## layer_norm (bfloat16)

candidate: hipbridge.kernels.norm.layer_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       291.4 [291.2-291.5]           4.0 [3.9-4.2]          9.6 [9.5-10.0]        20.6 [20.5-20.8]        14.1x         0.2x         0.5x  (latency-bound)
   1024x1024       317.2 [317.2-317.3]           5.7 [5.7-5.7]        10.0 [10.0-10.4]        22.9 [22.9-23.3]        13.9x         0.2x         0.4x  (latency-bound)
   4096x4096    1487.2 [1487.0-1487.3]        32.2 [31.9-32.8]        29.6 [28.6-30.3]        22.7 [21.7-22.7]        65.6x         1.4x         1.3x
```

## layer_norm_affine (float32)

candidate: hipbridge.kernels.norm.layer_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       208.7 [208.6-208.7]           3.9 [3.9-4.0]           8.6 [8.6-8.8]        25.8 [24.3-26.2]         8.1x         0.2x         0.3x  (latency-bound)
   1024x1024       241.0 [240.9-241.1]           5.5 [5.5-5.5]           9.7 [9.4-9.8]        25.7 [25.2-26.5]         9.4x         0.2x         0.4x  (latency-bound)
   4096x4096    1829.0 [1822.6-1837.1]        46.6 [46.5-46.8]        45.3 [44.5-45.3]        33.6 [33.5-33.8]        54.5x         1.4x         1.3x
```

## layer_norm_affine (float16)

candidate: hipbridge.kernels.norm.layer_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       276.0 [275.9-276.1]           4.0 [4.0-4.0]           8.6 [8.4-8.6]        23.3 [23.1-25.5]        11.8x         0.2x         0.4x  (latency-bound)
   1024x1024       306.4 [306.3-306.4]           5.7 [5.7-5.7]         10.6 [9.9-11.3]        26.3 [25.8-26.7]        11.6x         0.2x         0.4x  (latency-bound)
   4096x4096    1423.6 [1423.6-1423.9]        33.7 [33.3-34.1]        29.7 [29.6-32.1]        22.8 [22.7-23.1]        62.5x         1.5x         1.3x
```

## layer_norm_affine (bfloat16)

candidate: hipbridge.kernels.norm.layer_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       301.2 [301.1-301.3]           4.1 [4.1-4.1]           8.7 [8.7-8.9]        23.2 [23.0-23.2]        13.0x         0.2x         0.4x  (latency-bound)
   1024x1024       332.1 [331.7-332.5]           5.8 [5.8-5.8]           8.7 [8.7-9.3]        23.3 [23.3-23.9]        14.2x         0.2x         0.4x  (latency-bound)
   4096x4096    1579.8 [1579.5-1580.8]        35.6 [35.1-36.7]        31.7 [31.1-32.7]        23.0 [23.0-23.5]        68.6x         1.5x         1.4x
```

## rms_norm (float32)

candidate: hipbridge.kernels.norm.rms_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       169.8 [169.7-169.8]           2.8 [2.8-2.9]           7.8 [7.8-8.0]        21.7 [20.6-22.7]         7.8x         0.1x         0.4x  (latency-bound)
   1024x1024       193.8 [193.7-193.9]           4.0 [4.0-4.1]           7.6 [7.3-8.3]        18.9 [18.9-19.1]        10.2x         0.2x         0.4x  (latency-bound)
   4096x4096    1443.3 [1438.0-1460.9]        41.2 [41.1-41.4]        42.9 [42.7-43.1]        31.4 [31.3-31.5]        46.0x         1.3x         1.4x
```

## rms_norm (float16)

candidate: hipbridge.kernels.norm.rms_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       202.9 [202.8-202.9]           2.9 [2.8-2.9]           7.3 [7.3-7.4]        18.4 [18.4-18.6]        11.0x         0.2x         0.4x  (latency-bound)
   1024x1024       227.9 [227.7-227.9]           4.1 [4.1-4.1]           7.7 [7.6-7.8]        21.0 [20.1-21.9]        10.8x         0.2x         0.4x  (latency-bound)
   4096x4096    1053.7 [1053.0-1055.1]        22.9 [22.8-22.9]        19.9 [19.6-20.3]        18.6 [18.6-18.7]        56.6x         1.2x         1.1x
```

## rms_norm (bfloat16)

candidate: hipbridge.kernels.norm.rms_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       220.5 [220.5-220.5]           2.8 [2.8-2.9]           7.8 [7.3-8.2]        18.6 [18.5-18.7]        11.8x         0.2x         0.4x  (latency-bound)
   1024x1024       245.0 [244.9-245.4]           4.2 [4.2-4.2]           8.2 [8.0-9.1]        20.7 [20.4-20.8]        11.8x         0.2x         0.4x  (latency-bound)
   4096x4096    1145.5 [1145.4-1146.2]        24.9 [24.8-25.1]        21.6 [20.4-21.6]        18.5 [18.4-19.0]        61.9x         1.3x         1.2x
```

## rms_norm_affine (float32)

candidate: hipbridge.kernels.norm.rms_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       173.5 [173.4-173.6]           2.8 [2.8-2.9]           7.4 [7.2-7.5]        21.8 [20.6-21.8]         8.0x         0.1x         0.3x  (latency-bound)
   1024x1024       200.4 [200.4-200.5]           4.0 [4.0-4.1]           7.3 [7.2-7.6]        20.8 [20.7-20.8]         9.7x         0.2x         0.4x  (latency-bound)
   4096x4096    1583.7 [1580.8-1588.8]        41.4 [41.3-41.5]        43.3 [43.2-43.5]        32.3 [32.2-32.3]        49.1x         1.3x         1.3x
```

## rms_norm_affine (float16)

candidate: hipbridge.kernels.norm.rms_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       214.6 [214.4-214.6]           2.9 [2.8-2.9]           7.6 [7.4-7.9]        23.2 [20.7-23.4]         9.3x         0.1x         0.3x  (latency-bound)
   1024x1024       239.7 [239.0-239.8]           4.2 [4.2-4.2]           7.5 [7.4-7.8]        20.8 [20.5-20.8]        11.5x         0.2x         0.4x  (latency-bound)
   4096x4096    1129.5 [1129.1-1129.7]        25.5 [25.0-25.8]        21.8 [20.9-21.9]        20.6 [20.5-21.7]        54.8x         1.2x         1.1x
```

## rms_norm_affine (bfloat16)

candidate: hipbridge.kernels.norm.rms_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       233.8 [233.7-233.9]           2.9 [2.9-2.9]           7.2 [7.0-7.2]        21.0 [20.7-21.5]        11.1x         0.1x         0.3x  (latency-bound)
   1024x1024       259.8 [259.8-259.9]           4.2 [4.2-4.3]           7.4 [7.3-8.0]        20.8 [20.8-20.9]        12.5x         0.2x         0.4x  (latency-bound)
   4096x4096    1246.2 [1245.8-1246.2]        26.3 [26.2-26.3]        22.6 [21.6-23.1]        20.6 [20.6-20.7]        60.4x         1.3x         1.1x
```

## rope (float32)

candidate: hipbridge.kernels.rope (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024          80.5 [80.4-80.5]           2.2 [2.2-2.2]        46.7 [46.7-47.1]        24.8 [24.6-28.3]         3.3x         0.1x         1.9x  (latency-bound)
   1024x1024       102.1 [102.0-102.1]           2.8 [2.8-2.8]        48.1 [47.7-48.7]        24.8 [24.6-25.2]         4.1x         0.1x         1.9x  (latency-bound)
   4096x4096    1373.5 [1373.2-1373.5]        45.7 [45.6-46.3]     237.9 [237.8-238.9]        52.9 [52.7-52.9]        26.0x         0.9x         4.5x
```

## rope (float16)

candidate: hipbridge.kernels.rope (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024          79.5 [79.5-79.6]           2.2 [2.2-2.5]        47.4 [45.9-50.2]        25.8 [24.5-28.1]         3.1x         0.1x         1.8x  (latency-bound)
   1024x1024          97.1 [97.1-97.2]           2.7 [2.7-2.7]        48.5 [48.4-48.7]        25.0 [24.9-25.0]         3.9x         0.1x         1.9x  (latency-bound)
   4096x4096       735.2 [734.3-735.8]        24.0 [23.5-24.0]     140.3 [140.3-142.6]        26.5 [26.5-27.0]        27.7x         0.9x         5.3x
```

## rope (bfloat16)

candidate: hipbridge.kernels.rope (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       109.7 [109.6-109.7]           2.2 [2.2-2.2]        46.7 [46.6-47.3]        26.7 [26.5-27.8]         4.1x         0.1x         1.8x  (latency-bound)
   1024x1024       127.3 [127.3-127.3]           2.8 [2.8-2.8]        48.1 [48.1-53.6]        24.6 [24.4-25.0]         5.2x         0.1x         2.0x  (latency-bound)
   4096x4096       930.4 [930.4-933.2]        23.6 [23.6-24.2]     140.3 [140.0-142.6]        27.2 [27.1-28.7]        34.2x         0.9x         5.2x
```

## rms_norm_rope (float32)

candidate: hipbridge.kernels.fused.rms_norm_rope (Triton, fused)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)         2 launches (us)          candidate (us)  vs original vs tuned HIPvs 2 launches
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       120.8 [120.7-120.8]           2.8 [2.8-2.8]        47.9 [47.7-48.2]        27.8 [27.5-27.8]         4.3x         0.1x         1.7x  (latency-bound)
   1024x1024       140.4 [140.2-140.8]           4.3 [4.3-4.3]        52.3 [50.0-55.8]        28.0 [27.9-31.1]         5.0x         0.2x         1.9x  (latency-bound)
   4096x4096    2137.0 [2130.8-2140.3]        58.5 [58.5-58.7]        86.7 [86.1-87.8]        71.3 [70.4-72.1]        30.0x         0.8x         1.2x
```

## rms_norm_rope (float16)

candidate: hipbridge.kernels.fused.rms_norm_rope (Triton, fused)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)         2 launches (us)          candidate (us)  vs original vs tuned HIPvs 2 launches
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       151.5 [151.3-151.6]           2.6 [2.6-2.6]        55.0 [53.8-55.4]        30.7 [30.6-30.8]         4.9x         0.1x         1.8x  (latency-bound)
   1024x1024       170.2 [169.7-170.4]           4.1 [4.1-4.1]        55.2 [54.3-56.9]        29.1 [28.5-30.9]         5.9x         0.1x         1.9x  (latency-bound)
   4096x4096    1169.4 [1165.8-1170.3]        29.7 [29.6-30.1]        59.4 [58.7-61.1]        40.7 [40.4-40.8]        28.8x         0.7x         1.5x
```

## rms_norm_rope (bfloat16)

candidate: hipbridge.kernels.fused.rms_norm_rope (Triton, fused)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)         2 launches (us)          candidate (us)  vs original vs tuned HIPvs 2 launches
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       183.8 [183.8-183.9]           2.7 [2.7-2.8]        50.1 [49.1-56.1]        31.2 [31.2-31.4]         5.9x         0.1x         1.6x  (latency-bound)
   1024x1024       202.6 [202.6-202.8]           4.2 [4.2-4.2]        54.2 [54.2-55.3]        31.2 [30.9-32.4]         6.5x         0.1x         1.7x  (latency-bound)
   4096x4096    1356.1 [1355.1-1357.8]        29.9 [29.7-30.0]        54.1 [53.6-54.2]        41.0 [40.5-41.0]        33.1x         0.7x         1.3x
```

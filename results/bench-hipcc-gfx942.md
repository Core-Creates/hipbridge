# hipbridge benchmark

| field | value |
|---|---|
| generated | 2026-09-11 16:05 UTC |
| hipbridge | 0.1.0.dev0 at commit `8806003` |
| host | 2 (Linux x86_64) |
| device | AMD Radeon Graphics |
| toolchain | hipcc, HIP version: 7.14.60850-0000000 |
| arch | gfx942 |
| torch | 2.9.1+rocm6.4 |
| measured code | `c572bb7c221e07b1` |

## row_softmax (float32)

candidate: hipbridge.kernels.softmax (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       324.2 [324.0-325.3]           4.0 [4.0-4.1]           5.4 [5.4-5.6]        20.2 [20.2-21.4]        16.1x         0.2x         0.3x  (latency-bound)
   1024x1024       363.5 [363.4-364.2]           6.3 [6.3-6.3]           5.3 [5.2-5.6]        18.2 [18.1-18.5]        20.0x         0.3x         0.3x  (latency-bound)
   4096x4096    2429.8 [2429.4-2430.3]        52.3 [52.1-53.1]        40.1 [39.4-40.7]        32.6 [32.4-32.7]        74.6x         1.6x         1.2x
```

## row_softmax (float16)

candidate: hipbridge.kernels.softmax (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       309.0 [308.9-309.1]           4.0 [3.9-4.0]           5.5 [5.3-5.5]        18.2 [18.2-18.4]        17.0x         0.2x         0.3x  (latency-bound)
   1024x1024       342.1 [341.9-342.2]           6.4 [6.4-6.4]           6.7 [6.7-6.7]        20.1 [19.2-21.0]        17.0x         0.3x         0.3x  (latency-bound)
   4096x4096    2746.3 [2745.9-2746.6]        44.6 [44.3-45.6]        23.8 [23.6-25.3]        23.9 [23.6-24.1]       114.9x         1.9x         1.0x
```

## row_softmax (bfloat16)

candidate: hipbridge.kernels.softmax (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       339.3 [339.3-339.5]           4.0 [4.0-4.0]           5.6 [5.6-5.6]        17.9 [17.8-18.3]        19.0x         0.2x         0.3x  (latency-bound)
   1024x1024       372.5 [372.5-372.6]           6.5 [6.5-6.6]           7.2 [7.2-7.2]        20.0 [19.1-20.7]        18.6x         0.3x         0.4x  (latency-bound)
   4096x4096    3228.0 [3226.3-3229.1]        46.1 [45.5-46.3]        26.1 [25.2-27.4]        25.5 [25.3-25.6]       126.5x         1.8x         1.0x
```

## layer_norm (float32)

candidate: hipbridge.kernels.norm.layer_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       205.0 [205.0-205.1]           3.9 [3.9-4.1]           8.1 [8.0-8.2]        20.1 [18.8-20.4]        10.2x         0.2x         0.4x  (latency-bound)
   1024x1024       233.6 [233.5-233.6]           5.4 [5.4-5.4]           9.3 [9.3-9.6]        21.0 [20.1-23.2]        11.1x         0.3x         0.4x  (latency-bound)
   4096x4096    1735.9 [1733.7-1740.7]        45.9 [45.8-46.0]        44.7 [44.6-45.5]        31.3 [31.2-31.3]        55.5x         1.5x         1.4x
```

## layer_norm (float16)

candidate: hipbridge.kernels.norm.layer_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       272.1 [271.9-272.1]           3.9 [3.9-4.1]          9.8 [9.7-10.5]        18.8 [18.8-18.8]        14.5x         0.2x         0.5x  (latency-bound)
   1024x1024       297.2 [297.1-297.2]           5.6 [5.6-5.6]           8.8 [8.7-9.1]        19.1 [18.6-22.4]        15.6x         0.3x         0.5x  (latency-bound)
   4096x4096    1370.5 [1370.3-1370.6]        30.8 [30.6-30.9]        27.4 [27.3-29.4]        20.8 [20.0-21.0]        65.9x         1.5x         1.3x
```

## layer_norm (bfloat16)

candidate: hipbridge.kernels.norm.layer_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       291.5 [291.1-291.7]           3.9 [3.9-4.0]           8.8 [8.6-8.9]        18.6 [18.5-18.9]        15.7x         0.2x         0.5x  (latency-bound)
   1024x1024       317.1 [317.0-317.3]           5.7 [5.6-5.7]           8.5 [8.3-9.1]        18.5 [18.4-18.8]        17.1x         0.3x         0.5x  (latency-bound)
   4096x4096    1483.9 [1483.5-1484.2]        31.8 [31.6-31.8]        29.2 [28.7-30.4]        19.1 [18.9-19.5]        77.8x         1.7x         1.5x
```

## layer_norm_affine (float32)

candidate: hipbridge.kernels.norm.layer_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       209.0 [208.9-209.2]           4.0 [3.9-4.0]           9.5 [9.4-9.9]        27.9 [27.3-27.9]         7.5x         0.1x         0.3x  (latency-bound)
   1024x1024       241.1 [240.9-241.2]           5.5 [5.5-5.6]           8.9 [8.5-9.1]        22.7 [22.6-23.4]        10.6x         0.2x         0.4x  (latency-bound)
   4096x4096    1852.8 [1845.2-1854.7]        46.6 [46.5-46.8]        45.5 [44.5-46.1]        33.6 [33.5-33.6]        55.2x         1.4x         1.4x
```

## layer_norm_affine (float16)

candidate: hipbridge.kernels.norm.layer_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       275.8 [275.7-275.9]           4.1 [4.0-4.1]           8.5 [8.5-8.7]        23.3 [22.9-23.9]        11.9x         0.2x         0.4x  (latency-bound)
   1024x1024       305.8 [305.7-306.0]           5.7 [5.7-5.8]           8.6 [8.6-9.1]        23.2 [23.0-23.4]        13.2x         0.2x         0.4x  (latency-bound)
   4096x4096    1421.6 [1420.8-1422.0]        33.5 [33.3-34.3]        30.2 [29.7-32.4]        23.1 [22.9-23.7]        61.6x         1.5x         1.3x
```

## layer_norm_affine (bfloat16)

candidate: hipbridge.kernels.norm.layer_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       301.4 [301.4-301.4]           4.0 [4.0-4.0]           9.2 [8.5-9.3]        26.5 [26.3-26.6]        11.4x         0.2x         0.3x  (latency-bound)
   1024x1024       332.4 [332.2-332.7]           5.7 [5.7-5.8]           8.7 [8.6-8.9]        23.2 [22.8-23.4]        14.3x         0.2x         0.4x  (latency-bound)
   4096x4096    1581.6 [1581.4-1582.6]        35.5 [35.0-35.6]        31.8 [31.1-32.4]        23.3 [23.1-23.4]        68.0x         1.5x         1.4x
```

## rms_norm (float32)

candidate: hipbridge.kernels.norm.rms_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       169.7 [169.7-169.8]           2.7 [2.7-2.7]           7.8 [7.4-7.8]        18.8 [18.7-18.8]         9.0x         0.1x         0.4x  (latency-bound)
   1024x1024       193.6 [193.6-193.6]           4.0 [4.0-4.1]           8.8 [8.8-9.3]        23.0 [22.8-23.2]         8.4x         0.2x         0.4x  (latency-bound)
   4096x4096    1451.7 [1446.2-1461.2]        41.3 [41.1-41.4]        43.2 [42.8-43.3]        31.5 [31.4-31.5]        46.1x         1.3x         1.4x
```

## rms_norm (float16)

candidate: hipbridge.kernels.norm.rms_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       202.5 [202.4-202.6]           2.7 [2.7-2.7]           7.4 [7.3-7.6]        18.5 [18.3-18.7]        10.9x         0.1x         0.4x  (latency-bound)
   1024x1024       227.8 [227.7-227.9]           4.1 [4.1-4.1]           8.6 [8.2-8.6]        20.6 [20.5-20.7]        11.1x         0.2x         0.4x  (latency-bound)
   4096x4096    1058.2 [1057.2-1058.8]        22.8 [22.6-23.2]        20.1 [19.4-20.2]        18.8 [18.5-19.1]        56.2x         1.2x         1.1x
```

## rms_norm (bfloat16)

candidate: hipbridge.kernels.norm.rms_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       220.1 [220.0-220.2]           2.8 [2.8-2.8]           7.4 [7.3-7.5]        18.7 [18.4-18.8]        11.8x         0.1x         0.4x  (latency-bound)
   1024x1024       245.2 [245.2-245.3]           4.2 [4.2-4.2]           8.6 [8.4-8.8]        22.1 [20.5-23.2]        11.1x         0.2x         0.4x  (latency-bound)
   4096x4096    1141.2 [1141.2-1141.7]        24.5 [24.4-24.7]        21.6 [20.4-21.7]        19.8 [19.5-20.4]        57.5x         1.2x         1.1x
```

## rms_norm_affine (float32)

candidate: hipbridge.kernels.norm.rms_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       173.7 [173.5-173.7]           2.9 [2.9-2.9]           7.7 [7.2-8.2]        22.4 [21.2-23.5]         7.8x         0.1x         0.3x  (latency-bound)
   1024x1024       200.5 [200.3-200.5]           4.0 [4.0-4.1]           8.1 [7.8-8.3]        22.8 [22.7-23.2]         8.8x         0.2x         0.4x  (latency-bound)
   4096x4096    1585.9 [1573.9-1595.2]        41.5 [41.4-41.5]        43.4 [43.2-43.5]        32.1 [32.0-32.2]        49.4x         1.3x         1.3x
```

## rms_norm_affine (float16)

candidate: hipbridge.kernels.norm.rms_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       214.5 [214.3-214.5]           2.8 [2.8-2.8]           7.6 [7.3-7.8]        21.1 [20.7-21.2]        10.1x         0.1x         0.4x  (latency-bound)
   1024x1024       239.7 [239.5-239.7]           4.2 [4.2-4.2]           7.5 [7.3-7.8]        20.8 [20.7-21.1]        11.5x         0.2x         0.4x  (latency-bound)
   4096x4096    1124.4 [1123.9-1125.2]        25.2 [25.1-25.2]        21.4 [20.8-21.6]        21.1 [21.0-21.1]        53.4x         1.2x         1.0x
```

## rms_norm_affine (bfloat16)

candidate: hipbridge.kernels.norm.rms_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       234.0 [233.9-234.1]           3.0 [2.9-3.0]           7.3 [7.1-7.7]        20.4 [20.3-20.6]        11.5x         0.1x         0.4x  (latency-bound)
   1024x1024       259.8 [259.6-259.8]           4.3 [4.3-4.3]           7.2 [6.9-7.3]        21.0 [20.7-21.2]        12.4x         0.2x         0.3x  (latency-bound)
   4096x4096    1247.2 [1247.1-1247.9]        26.2 [26.2-26.3]        23.3 [21.6-23.5]        21.3 [21.3-21.4]        58.5x         1.2x         1.1x
```

## rope (float32)

candidate: hipbridge.kernels.rope (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024          80.6 [80.5-80.6]           2.2 [2.1-2.3]        49.5 [48.8-49.8]        24.5 [24.5-24.6]         3.3x         0.1x         2.0x  (latency-bound)
   1024x1024       101.9 [101.8-102.0]           2.8 [2.8-2.8]        51.1 [47.6-51.1]        27.1 [24.4-27.5]         3.8x         0.1x         1.9x  (latency-bound)
   4096x4096    1373.6 [1372.4-1373.7]        45.9 [45.9-46.1]     238.7 [238.4-240.2]        52.7 [52.7-52.8]        26.0x         0.9x         4.5x
```

## rope (float16)

candidate: hipbridge.kernels.rope (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024          79.5 [79.5-79.6]           2.2 [2.1-2.3]        48.7 [47.5-48.8]        24.2 [24.0-24.3]         3.3x         0.1x         2.0x  (latency-bound)
   1024x1024          97.1 [97.0-97.1]           2.7 [2.7-2.9]        48.2 [47.0-48.6]        25.4 [24.5-27.2]         3.8x         0.1x         1.9x  (latency-bound)
   4096x4096       742.0 [741.3-809.3]        23.5 [23.4-23.8]     138.6 [138.5-140.3]        27.4 [26.9-27.4]        27.1x         0.9x         5.1x
```

## rope (bfloat16)

candidate: hipbridge.kernels.rope (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       109.7 [109.7-109.7]           2.2 [2.1-2.2]        47.7 [46.7-50.1]        24.4 [24.3-24.5]         4.5x         0.1x         2.0x  (latency-bound)
   1024x1024       127.4 [127.3-127.4]           2.8 [2.8-2.8]        48.8 [47.1-49.3]        24.1 [24.0-24.2]         5.3x         0.1x         2.0x  (latency-bound)
   4096x4096       934.2 [931.8-945.2]        23.6 [23.5-23.7]     139.9 [138.4-141.8]        27.6 [27.5-27.8]        33.9x         0.9x         5.1x
```

## rms_norm_rope (float32)

candidate: hipbridge.kernels.fused.rms_norm_rope (Triton, fused)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)         2 launches (us)          candidate (us)  vs original vs tuned HIPvs 2 launches
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       120.6 [120.6-120.6]           2.8 [2.8-2.8]        49.4 [49.0-49.4]        27.6 [27.2-27.7]         4.4x         0.1x         1.8x  (latency-bound)
   1024x1024       140.4 [140.2-140.7]           4.3 [4.3-4.4]        49.2 [48.5-49.3]        27.5 [27.4-28.2]         5.1x         0.2x         1.8x  (latency-bound)
   4096x4096    2139.2 [2138.6-2144.2]        58.4 [58.3-58.6]        87.3 [86.6-88.4]        71.8 [70.9-73.7]        29.8x         0.8x         1.2x
```

## rms_norm_rope (float16)

candidate: hipbridge.kernels.fused.rms_norm_rope (Triton, fused)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)         2 launches (us)          candidate (us)  vs original vs tuned HIPvs 2 launches
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       151.6 [151.5-151.6]           2.7 [2.7-2.7]        52.7 [47.9-54.9]        27.5 [27.5-27.6]         5.5x         0.1x         1.9x  (latency-bound)
   1024x1024       169.7 [169.6-170.1]           4.1 [4.1-4.1]        48.1 [48.0-48.4]        27.6 [27.6-28.0]         6.1x         0.1x         1.7x  (latency-bound)
   4096x4096    1208.6 [1167.8-1231.4]        29.9 [29.5-29.9]        51.1 [50.8-54.9]        41.0 [41.0-42.0]        29.5x         0.7x         1.2x
```

## rms_norm_rope (bfloat16)

candidate: hipbridge.kernels.fused.rms_norm_rope (Triton, fused)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)         2 launches (us)          candidate (us)  vs original vs tuned HIPvs 2 launches
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       183.7 [183.7-183.8]           2.8 [2.8-2.8]        49.1 [48.9-55.0]        27.7 [27.6-27.9]         6.6x         0.1x         1.8x  (latency-bound)
   1024x1024       202.8 [202.7-203.3]           4.2 [4.2-4.2]        54.0 [53.3-54.2]        28.0 [27.9-28.8]         7.2x         0.1x         1.9x  (latency-bound)
   4096x4096    1355.3 [1354.6-1355.7]        29.9 [29.6-30.1]        48.6 [48.0-49.5]        41.2 [40.7-42.6]        32.9x         0.7x         1.2x
```

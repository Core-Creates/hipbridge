# hipbridge benchmark

| field | value |
|---|---|
| generated | 2026-09-12 03:49 UTC |
| hipbridge | 0.1.0.dev0 at commit `bad9fab` |
| host | 2 (Linux x86_64) |
| device | AMD Radeon Graphics |
| toolchain | hipcc, HIP version: 7.14.60850-0000000 |
| arch | gfx942 |
| torch | 2.9.1+rocm6.4 |
| measured code | `c2d376c8a380e3ee` |

## row_softmax (float32)

candidate: hipbridge.kernels.softmax (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       323.8 [323.6-324.1]           4.1 [4.0-4.1]           5.0 [4.9-5.2]        18.0 [17.9-18.1]        18.0x         0.2x         0.3x  (latency-bound)
   1024x1024       363.9 [363.7-364.3]           6.3 [6.3-6.3]           5.2 [5.2-5.4]        17.7 [17.7-17.8]        20.5x         0.4x         0.3x  (latency-bound)
   4096x4096    2425.3 [2425.1-2426.3]        53.5 [53.3-53.8]        39.1 [38.7-40.3]        32.6 [32.4-32.6]        74.3x         1.6x         1.2x
```

## row_softmax (float16)

candidate: hipbridge.kernels.softmax (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       308.0 [307.8-308.5]           3.9 [3.9-4.0]           5.2 [5.1-5.3]        17.9 [17.8-18.3]        17.2x         0.2x         0.3x  (latency-bound)
   1024x1024       341.8 [341.8-342.2]           6.4 [6.4-6.4]           6.7 [6.7-6.7]        21.9 [18.2-25.2]        15.6x         0.3x         0.3x  (latency-bound)
   4096x4096    2744.3 [2742.4-2744.4]        45.1 [44.6-45.2]        24.7 [23.5-26.2]        24.0 [23.8-25.0]       114.2x         1.9x         1.0x
```

## row_softmax (bfloat16)

candidate: hipbridge.kernels.softmax (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       339.3 [339.1-339.4]           4.1 [4.0-4.1]           5.6 [5.6-5.7]        20.2 [20.1-21.3]        16.8x         0.2x         0.3x  (latency-bound)
   1024x1024       372.6 [372.3-372.8]           6.6 [6.5-6.6]           7.2 [7.1-7.2]        18.2 [18.2-18.2]        20.5x         0.4x         0.4x  (latency-bound)
   4096x4096    3224.0 [3222.4-3225.8]        45.9 [45.7-46.1]        26.7 [25.3-26.9]        25.7 [25.5-25.8]       125.4x         1.8x         1.0x
```

## layer_norm (float32)

candidate: hipbridge.kernels.norm.layer_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       205.1 [205.0-205.2]           3.9 [3.9-3.9]           8.0 [7.8-8.0]        18.5 [18.3-18.9]        11.1x         0.2x         0.4x  (latency-bound)
   1024x1024       233.5 [233.5-233.6]           5.4 [5.4-5.4]           8.0 [8.0-8.5]        19.3 [18.6-27.6]        12.1x         0.3x         0.4x  (latency-bound)
   4096x4096    1745.5 [1742.0-1760.7]        45.8 [45.8-46.3]        44.9 [44.6-45.4]        31.2 [31.1-31.4]        56.0x         1.5x         1.4x
```

## layer_norm (float16)

candidate: hipbridge.kernels.norm.layer_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       272.1 [272.1-272.3]           4.0 [3.8-4.1]           8.2 [8.2-8.4]        18.6 [18.5-18.9]        14.6x         0.2x         0.4x  (latency-bound)
   1024x1024       297.2 [296.9-297.3]           5.6 [5.6-5.6]           8.7 [8.6-8.7]        19.0 [18.8-19.0]        15.7x         0.3x         0.5x  (latency-bound)
   4096x4096    1371.9 [1371.4-1372.2]        30.7 [30.5-30.8]        28.3 [27.2-29.3]        18.7 [18.3-18.7]        73.5x         1.6x         1.5x
```

## layer_norm (bfloat16)

candidate: hipbridge.kernels.norm.layer_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       291.2 [291.2-291.4]           3.9 [3.9-4.0]           8.3 [8.2-8.4]        18.6 [18.4-18.8]        15.6x         0.2x         0.4x  (latency-bound)
   1024x1024       317.1 [317.0-317.1]           5.6 [5.6-5.7]           8.7 [8.4-8.8]        18.6 [18.5-18.6]        17.0x         0.3x         0.5x  (latency-bound)
   4096x4096    1482.6 [1482.5-1483.1]        31.7 [31.2-31.8]        29.2 [28.4-29.9]        18.6 [18.6-18.9]        79.5x         1.7x         1.6x
```

## layer_norm_affine (float32)

candidate: hipbridge.kernels.norm.layer_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       209.0 [208.9-209.0]           3.9 [3.8-4.0]           8.2 [8.1-8.3]        22.8 [22.7-23.0]         9.2x         0.2x         0.4x  (latency-bound)
   1024x1024       240.9 [240.8-241.1]           5.5 [5.5-5.5]           8.3 [8.3-8.5]        23.0 [22.8-23.1]        10.5x         0.2x         0.4x  (latency-bound)
   4096x4096    1844.5 [1838.9-1856.2]        46.8 [46.7-47.6]        45.5 [44.8-45.8]        33.6 [33.4-33.6]        54.9x         1.4x         1.4x
```

## layer_norm_affine (float16)

candidate: hipbridge.kernels.norm.layer_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       275.8 [275.8-276.0]           4.0 [4.0-4.1]        10.3 [10.2-10.3]        27.0 [25.7-27.8]        10.2x         0.1x         0.4x  (latency-bound)
   1024x1024       305.7 [305.6-306.4]           5.7 [5.7-5.7]           8.7 [8.6-8.8]        23.5 [23.1-23.7]        13.0x         0.2x         0.4x  (latency-bound)
   4096x4096    1422.2 [1420.4-1422.5]        33.5 [33.4-34.3]        30.4 [30.2-32.8]        23.4 [23.2-23.4]        60.8x         1.4x         1.3x
```

## layer_norm_affine (bfloat16)

candidate: hipbridge.kernels.norm.layer_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       301.2 [301.2-301.3]           4.0 [4.0-4.1]           9.1 [8.8-9.1]        23.1 [22.7-24.6]        13.1x         0.2x         0.4x  (latency-bound)
   1024x1024       332.2 [331.9-332.6]           5.7 [5.7-5.8]           8.6 [8.5-8.6]        23.2 [22.9-23.3]        14.3x         0.2x         0.4x  (latency-bound)
   4096x4096    1583.3 [1583.1-1584.2]        35.7 [35.2-36.3]        31.6 [31.1-32.4]        28.5 [28.2-29.0]        55.5x         1.3x         1.1x
```

## rms_norm (float32)

candidate: hipbridge.kernels.norm.rms_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       169.8 [169.7-170.0]           2.8 [2.8-2.8]           7.2 [7.1-7.2]        18.7 [18.6-18.9]         9.1x         0.1x         0.4x  (latency-bound)
   1024x1024       193.6 [193.6-193.8]           4.0 [4.0-4.0]           7.2 [7.1-7.4]        18.5 [18.4-18.5]        10.5x         0.2x         0.4x  (latency-bound)
   4096x4096    1463.4 [1455.1-1469.5]        41.2 [41.2-41.3]        42.7 [42.7-42.9]        31.4 [31.2-31.5]        46.5x         1.3x         1.4x
```

## rms_norm (float16)

candidate: hipbridge.kernels.norm.rms_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       202.8 [202.5-202.9]           2.7 [2.7-2.7]           7.0 [6.9-7.2]        18.7 [18.6-18.8]        10.8x         0.1x         0.4x  (latency-bound)
   1024x1024       227.6 [227.0-227.6]           4.1 [4.1-4.2]           7.3 [7.3-7.5]        18.8 [18.6-19.3]        12.1x         0.2x         0.4x  (latency-bound)
   4096x4096    1054.2 [1053.3-1054.7]        22.8 [22.6-23.1]        19.8 [19.6-20.0]        18.6 [18.5-18.6]        56.8x         1.2x         1.1x
```

## rms_norm (bfloat16)

candidate: hipbridge.kernels.norm.rms_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       220.3 [220.3-220.4]           2.8 [2.8-2.8]           7.0 [6.9-7.4]        18.5 [18.2-18.6]        11.9x         0.2x         0.4x  (latency-bound)
   1024x1024       245.3 [245.1-245.5]           4.2 [4.2-4.2]           8.7 [8.6-8.8]        21.1 [20.5-21.9]        11.6x         0.2x         0.4x  (latency-bound)
   4096x4096    1147.1 [1146.4-1147.1]        24.6 [24.6-24.9]        21.2 [20.4-21.5]        18.6 [18.4-18.7]        61.7x         1.3x         1.1x
```

## rms_norm_affine (float32)

candidate: hipbridge.kernels.norm.rms_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       173.8 [173.8-173.9]           2.9 [2.9-2.9]           6.8 [6.7-6.8]        20.8 [20.7-20.8]         8.4x         0.1x         0.3x  (latency-bound)
   1024x1024       200.3 [200.2-200.4]           4.0 [4.0-4.1]           7.3 [7.1-7.7]        21.1 [21.1-21.2]         9.5x         0.2x         0.3x  (latency-bound)
   4096x4096    1584.1 [1580.2-1587.7]        41.4 [41.2-41.5]        43.0 [43.0-43.3]        32.2 [32.2-32.6]        49.2x         1.3x         1.3x
```

## rms_norm_affine (float16)

candidate: hipbridge.kernels.norm.rms_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       214.2 [214.2-214.4]           2.8 [2.8-2.8]           7.1 [6.9-7.2]        20.8 [20.8-21.0]        10.3x         0.1x         0.3x  (latency-bound)
   1024x1024       239.2 [239.1-239.7]           4.2 [4.2-4.2]           7.4 [7.4-7.9]        21.2 [21.1-21.2]        11.3x         0.2x         0.4x  (latency-bound)
   4096x4096    1129.7 [1129.0-1130.6]        24.8 [24.8-25.5]        21.1 [20.7-21.7]        21.0 [21.0-21.2]        53.8x         1.2x         1.0x
```

## rms_norm_affine (bfloat16)

candidate: hipbridge.kernels.norm.rms_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       234.1 [233.9-234.1]           3.0 [3.0-3.0]           7.0 [6.9-7.0]        20.6 [20.5-20.8]        11.4x         0.1x         0.3x  (latency-bound)
   1024x1024       259.8 [259.8-259.8]           4.3 [4.2-4.3]           8.3 [8.1-8.6]        22.8 [22.6-22.9]        11.4x         0.2x         0.4x  (latency-bound)
   4096x4096    1244.8 [1244.8-1246.1]        26.1 [26.1-26.9]        22.6 [21.6-22.7]        21.1 [21.0-21.2]        59.0x         1.2x         1.1x
```

## rope (float32)

candidate: hipbridge.kernels.rope (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024          80.3 [80.3-80.4]           2.3 [2.2-2.4]        47.2 [46.0-47.9]        24.1 [24.0-24.4]         3.3x         0.1x         2.0x  (latency-bound)
   1024x1024       102.0 [102.0-102.1]           2.8 [2.8-2.8]        46.4 [46.3-47.1]        24.2 [24.1-24.3]         4.2x         0.1x         1.9x  (latency-bound)
   4096x4096    1373.2 [1372.7-1373.3]        45.8 [45.6-46.0]     239.3 [239.0-241.4]        53.6 [53.6-53.8]        25.6x         0.9x         4.5x
```

## rope (float16)

candidate: hipbridge.kernels.rope (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024          79.5 [79.5-79.6]           2.2 [2.1-2.2]        48.5 [47.8-52.8]        26.8 [24.2-27.8]         3.0x         0.1x         1.8x  (latency-bound)
   1024x1024          97.1 [97.1-97.1]           2.7 [2.7-3.0]        45.7 [45.6-46.4]        23.9 [23.8-24.1]         4.1x         0.1x         1.9x  (latency-bound)
   4096x4096       740.5 [739.6-807.4]        23.8 [23.5-24.2]     136.1 [135.9-138.7]        27.6 [27.4-27.7]        26.8x         0.9x         4.9x
```

## rope (bfloat16)

candidate: hipbridge.kernels.rope (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       109.7 [109.7-109.7]           2.1 [2.1-2.2]        47.8 [46.5-48.3]        26.9 [26.8-27.3]         4.1x         0.1x         1.8x  (latency-bound)
   1024x1024       127.4 [127.3-127.4]           2.9 [2.8-3.0]        48.8 [47.2-51.0]        24.4 [24.1-24.6]         5.2x         0.1x         2.0x  (latency-bound)
   4096x4096       933.3 [931.3-934.4]        23.8 [23.7-23.8]     136.8 [136.4-138.5]        27.9 [27.6-28.1]        33.5x         0.9x         4.9x
```

## rms_norm_rope (float32)

candidate: hipbridge.kernels.fused.rms_norm_rope (Triton, fused)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)         2 launches (us)          candidate (us)  vs original vs tuned HIPvs 2 launches
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       120.6 [120.5-120.6]           2.7 [2.7-2.7]        48.9 [48.8-48.9]        27.8 [27.6-27.9]         4.3x         0.1x         1.8x  (latency-bound)
   1024x1024       140.4 [140.2-140.8]           4.3 [4.3-4.3]        48.6 [48.2-48.7]        27.6 [27.6-27.8]         5.1x         0.2x         1.8x  (latency-bound)
   4096x4096    2129.1 [2123.5-2146.2]        58.5 [58.5-58.7]        86.5 [86.1-87.1]        71.8 [71.3-73.5]        29.7x         0.8x         1.2x
```

## rms_norm_rope (float16)

candidate: hipbridge.kernels.fused.rms_norm_rope (Triton, fused)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)         2 launches (us)          candidate (us)  vs original vs tuned HIPvs 2 launches
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       151.7 [151.7-151.7]           2.6 [2.6-2.6]        49.3 [49.2-49.7]        28.0 [27.9-28.0]         5.4x         0.1x         1.8x  (latency-bound)
   1024x1024       169.9 [169.7-170.6]           4.1 [4.1-4.1]        48.1 [48.1-48.5]        27.7 [27.6-27.9]         6.1x         0.1x         1.7x  (latency-bound)
   4096x4096    1163.8 [1163.5-1223.7]        29.5 [29.4-30.0]        48.5 [48.5-49.1]        41.3 [40.6-41.9]        28.2x         0.7x         1.2x
```

## rms_norm_rope (bfloat16)

candidate: hipbridge.kernels.fused.rms_norm_rope (Triton, fused)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)         2 launches (us)          candidate (us)  vs original vs tuned HIPvs 2 launches
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       183.7 [183.6-183.7]           2.8 [2.8-2.8]        48.6 [48.4-48.9]        27.6 [27.4-27.7]         6.7x         0.1x         1.8x  (latency-bound)
   1024x1024       203.2 [202.7-203.2]           4.2 [4.2-4.2]        48.6 [48.0-48.7]        27.9 [27.6-28.0]         7.3x         0.2x         1.7x  (latency-bound)
   4096x4096    1357.7 [1357.2-1358.8]        30.3 [30.3-30.4]        49.9 [49.4-50.0]        41.2 [41.0-42.1]        33.0x         0.7x         1.2x
```

# hipbridge benchmark

| field | value |
|---|---|
| generated | 2026-09-08 22:24 UTC |
| hipbridge | 0.1.0.dev0 at commit `a8ca0a7` |
| host | 2 (Linux x86_64) |
| device | AMD Radeon Graphics |
| toolchain | hipcc, HIP version: 7.14.60850-0000000 |
| arch | gfx942 |
| torch | 2.9.1+rocm6.4 |
| measured code | `6ab57a4fba756495` |

## row_softmax (float32)

candidate: hipbridge.kernels.softmax (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       326.0 [326.0-326.2]           4.0 [4.0-4.0]           5.2 [4.8-9.5]        17.7 [17.6-18.1]        18.5x         0.2x         0.3x  (latency-bound)
   1024x1024       364.2 [363.8-364.4]           6.3 [6.3-6.3]           5.5 [5.4-5.6]        18.6 [18.3-18.7]        19.5x         0.3x         0.3x  (latency-bound)
   4096x4096    2432.5 [2431.9-2432.9]        52.2 [51.3-52.5]        39.2 [38.5-40.2]        32.6 [32.5-32.7]        74.5x         1.6x         1.2x
```

## row_softmax (float16)

candidate: hipbridge.kernels.softmax (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       310.3 [310.2-310.4]           4.1 [4.1-4.1]           5.6 [5.6-5.7]        17.4 [16.4-17.6]        17.8x         0.2x         0.3x  (latency-bound)
   1024x1024       341.8 [341.7-341.8]           6.4 [6.4-6.4]           6.7 [6.7-6.7]        16.3 [16.2-16.5]        20.9x         0.4x         0.4x  (latency-bound)
   4096x4096    2744.3 [2744.1-2745.0]        44.9 [44.7-46.1]        24.6 [23.7-24.8]        24.1 [23.6-24.2]       113.6x         1.9x         1.0x
```

## row_softmax (bfloat16)

candidate: hipbridge.kernels.softmax (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       339.9 [339.8-340.0]           4.1 [4.1-4.1]           5.6 [5.5-5.6]        15.9 [15.9-16.0]        21.4x         0.3x         0.3x  (latency-bound)
   1024x1024       372.4 [372.1-372.6]           6.6 [6.6-6.6]           7.2 [7.1-7.2]        16.2 [16.1-16.5]        23.0x         0.4x         0.4x  (latency-bound)
   4096x4096    3225.6 [3225.2-3226.1]        46.5 [46.4-47.0]        26.3 [25.2-26.4]        25.6 [25.4-26.1]       125.8x         1.8x         1.0x
```

## layer_norm (float32)

candidate: hipbridge.kernels.norm.layer_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       204.9 [204.9-205.0]           3.9 [3.8-3.9]           9.0 [8.7-9.3]        17.8 [17.5-18.0]        11.5x         0.2x         0.5x  (latency-bound)
   1024x1024       233.4 [233.4-233.5]           5.4 [5.4-5.4]         10.5 [9.8-13.0]        24.1 [20.8-27.5]         9.7x         0.2x         0.4x  (latency-bound)
   4096x4096    1746.0 [1735.4-1747.9]        45.8 [45.8-46.2]        44.8 [44.6-45.2]        31.1 [31.0-31.4]        56.1x         1.5x         1.4x
```

## layer_norm (float16)

candidate: hipbridge.kernels.norm.layer_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       271.9 [271.9-272.2]           4.0 [3.9-4.1]           8.2 [8.2-8.3]        16.3 [16.3-16.5]        16.7x         0.2x         0.5x  (latency-bound)
   1024x1024       297.2 [297.1-297.7]           5.6 [5.5-5.6]           8.5 [8.4-8.6]        16.5 [16.5-16.6]        18.0x         0.3x         0.5x  (latency-bound)
   4096x4096    1370.9 [1370.7-1372.1]        30.4 [30.3-30.9]        28.3 [27.5-29.2]        18.4 [17.5-18.8]        74.5x         1.7x         1.5x
```

## layer_norm (bfloat16)

candidate: hipbridge.kernels.norm.layer_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       291.3 [291.3-291.4]           4.0 [3.9-4.0]           8.3 [8.2-8.5]        16.4 [16.3-16.5]        17.7x         0.2x         0.5x  (latency-bound)
   1024x1024       317.2 [317.0-317.5]           5.7 [5.6-5.7]           8.3 [8.3-9.0]        16.7 [16.7-16.8]        19.0x         0.3x         0.5x  (latency-bound)
   4096x4096    1487.3 [1486.7-1487.8]        32.3 [31.7-32.4]        29.8 [28.7-30.2]        18.9 [18.7-19.6]        78.9x         1.7x         1.6x
```

## layer_norm_affine (float32)

candidate: hipbridge.kernels.norm.layer_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       208.9 [208.9-208.9]           4.0 [3.9-4.0]           8.2 [8.1-8.3]        19.5 [19.5-19.6]        10.7x         0.2x         0.4x  (latency-bound)
   1024x1024       240.9 [240.9-240.9]           5.5 [5.5-5.5]           8.2 [8.2-8.3]        19.6 [19.3-19.8]        12.3x         0.3x         0.4x  (latency-bound)
   4096x4096    1834.4 [1825.0-1841.1]        46.8 [46.6-47.2]        45.1 [44.7-45.6]        33.5 [33.4-34.0]        54.7x         1.4x         1.3x
```

## layer_norm_affine (float16)

candidate: hipbridge.kernels.norm.layer_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       275.9 [275.7-276.1]           4.0 [4.0-4.1]           9.2 [9.2-9.3]        20.5 [20.5-20.6]        13.5x         0.2x         0.4x  (latency-bound)
   1024x1024       306.1 [305.5-306.3]           5.7 [5.7-5.7]           9.4 [8.9-9.4]        19.6 [19.3-20.0]        15.6x         0.3x         0.5x  (latency-bound)
   4096x4096    1422.6 [1422.3-1422.7]        33.9 [33.6-34.7]        30.7 [29.8-32.3]        19.5 [19.3-19.6]        73.1x         1.7x         1.6x
```

## layer_norm_affine (bfloat16)

candidate: hipbridge.kernels.norm.layer_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       301.2 [301.1-301.3]           4.1 [4.0-4.2]           8.3 [8.2-8.4]        20.5 [19.0-21.9]        14.7x         0.2x         0.4x  (latency-bound)
   1024x1024       331.7 [331.6-331.7]           5.7 [5.7-5.7]           8.6 [8.2-9.3]        19.3 [19.3-20.4]        17.2x         0.3x         0.4x  (latency-bound)
   4096x4096    1583.1 [1582.8-1584.6]        35.1 [34.9-35.2]        30.7 [30.7-32.2]        20.9 [20.7-20.9]        75.9x         1.7x         1.5x
```

## rms_norm (float32)

candidate: hipbridge.kernels.norm.rms_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       169.8 [169.7-169.8]           2.8 [2.8-2.8]           8.2 [8.1-8.4]        18.4 [17.2-18.5]         9.2x         0.2x         0.4x  (latency-bound)
   1024x1024       193.6 [193.5-193.7]           4.0 [4.0-4.1]           7.9 [7.8-8.0]        18.8 [18.2-26.0]        10.3x         0.2x         0.4x  (latency-bound)
   4096x4096    1457.3 [1453.5-1460.5]        41.0 [40.9-41.3]        43.0 [43.0-43.2]        31.4 [31.3-31.4]        46.4x         1.3x         1.4x
```

## rms_norm (float16)

candidate: hipbridge.kernels.norm.rms_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       202.9 [202.8-202.9]           2.8 [2.8-2.8]           7.0 [6.9-7.2]        16.4 [16.3-18.1]        12.4x         0.2x         0.4x  (latency-bound)
   1024x1024       227.6 [227.2-228.0]           4.1 [4.1-4.1]           7.6 [7.1-7.8]        16.4 [16.2-16.9]        13.9x         0.2x         0.5x  (latency-bound)
   4096x4096    1058.3 [1058.0-1058.5]        22.9 [22.8-23.0]        20.1 [19.4-20.6]        17.0 [16.9-17.1]        62.4x         1.3x         1.2x
```

## rms_norm (bfloat16)

candidate: hipbridge.kernels.norm.rms_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       220.2 [220.2-220.4]           2.9 [2.9-2.9]           6.9 [6.9-7.1]        16.3 [16.3-16.3]        13.5x         0.2x         0.4x  (latency-bound)
   1024x1024       244.9 [244.9-245.3]           4.2 [4.2-4.2]           7.3 [7.3-7.4]        18.5 [17.8-18.9]        13.2x         0.2x         0.4x  (latency-bound)
   4096x4096    1144.0 [1143.5-1144.1]        24.8 [24.5-24.8]        21.7 [20.4-21.7]        17.0 [17.0-17.3]        67.2x         1.5x         1.3x
```

## rms_norm_affine (float32)

candidate: hipbridge.kernels.norm.rms_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       173.6 [173.5-173.7]           2.8 [2.8-2.8]           7.7 [7.5-7.7]        19.9 [19.5-19.9]         8.7x         0.1x         0.4x  (latency-bound)
   1024x1024       200.6 [200.4-200.6]           4.0 [4.0-4.0]           7.1 [7.1-7.3]        18.1 [18.1-18.6]        11.1x         0.2x         0.4x  (latency-bound)
   4096x4096    1569.5 [1568.1-1578.9]        41.2 [41.2-41.4]        43.3 [43.2-43.7]        32.3 [32.0-32.5]        48.6x         1.3x         1.3x
```

## rms_norm_affine (float16)

candidate: hipbridge.kernels.norm.rms_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       214.8 [214.6-214.9]           2.9 [2.9-2.9]           7.2 [7.1-7.3]        17.9 [17.7-17.9]        12.0x         0.2x         0.4x  (latency-bound)
   1024x1024       239.1 [238.9-239.2]           4.2 [4.2-4.2]           7.4 [7.3-7.5]        18.1 [18.0-18.2]        13.2x         0.2x         0.4x  (latency-bound)
   4096x4096    1129.7 [1129.6-1129.7]        25.1 [25.0-25.5]        21.4 [20.7-21.6]        17.9 [17.8-19.8]        63.2x         1.4x         1.2x
```

## rms_norm_affine (bfloat16)

candidate: hipbridge.kernels.norm.rms_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       234.1 [233.9-234.2]           3.0 [2.9-3.0]           7.5 [7.3-8.0]        19.4 [17.9-19.8]        12.0x         0.2x         0.4x  (latency-bound)
   1024x1024       259.7 [259.7-259.9]           4.2 [4.2-4.3]           7.2 [7.1-7.6]        18.3 [18.2-18.5]        14.2x         0.2x         0.4x  (latency-bound)
   4096x4096    1242.3 [1241.8-1242.7]        26.4 [26.3-26.9]        22.4 [21.5-23.0]        18.0 [17.8-19.3]        68.9x         1.5x         1.2x
```

## rope (float32)

candidate: hipbridge.kernels.rope (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024          80.5 [80.4-80.5]           2.3 [2.3-2.4]        48.2 [47.6-48.9]        22.6 [22.6-22.8]         3.6x         0.1x         2.1x  (latency-bound)
   1024x1024       101.9 [101.8-102.0]           2.8 [2.8-2.9]        51.4 [48.1-51.7]        21.2 [19.8-22.0]         4.8x         0.1x         2.4x  (latency-bound)
   4096x4096    1372.5 [1370.4-1372.5]        45.9 [45.4-46.0]     240.0 [240.0-241.9]        53.2 [53.2-53.3]        25.8x         0.9x         4.5x
```

## rope (float16)

candidate: hipbridge.kernels.rope (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024          79.5 [79.5-79.5]           2.2 [2.2-2.3]        60.4 [51.2-60.6]        24.4 [23.7-24.4]         3.3x         0.1x         2.5x  (latency-bound)
   1024x1024          97.1 [97.1-97.2]           2.7 [2.7-2.7]        51.3 [46.6-51.8]        20.0 [19.9-20.3]         4.9x         0.1x         2.6x  (latency-bound)
   4096x4096       738.4 [738.3-738.6]        24.0 [23.5-24.0]     135.3 [135.3-137.4]        27.7 [27.3-27.7]        26.7x         0.9x         4.9x
```

## rope (bfloat16)

candidate: hipbridge.kernels.rope (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       109.6 [109.6-109.7]           2.3 [2.2-2.3]        47.2 [46.5-48.1]        19.9 [19.9-20.2]         5.5x         0.1x         2.4x  (latency-bound)
   1024x1024       127.4 [127.3-127.4]           2.8 [2.8-2.9]        45.5 [45.4-46.3]        20.0 [19.9-20.0]         6.4x         0.1x         2.3x  (latency-bound)
   4096x4096       936.2 [934.8-937.3]        23.9 [23.6-24.0]     136.1 [135.7-138.2]        28.1 [27.5-28.2]        33.3x         0.8x         4.8x
```

## rms_norm_rope (float32)

candidate: hipbridge.kernels.fused.rms_norm_rope (Triton, fused)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)         2 launches (us)          candidate (us)  vs original vs tuned HIPvs 2 launches
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       120.7 [120.6-120.7]           2.8 [2.7-2.8]        39.5 [39.5-39.5]        22.0 [21.9-22.0]         5.5x         0.1x         1.8x  (latency-bound)
   1024x1024       140.4 [140.2-140.6]           4.3 [4.3-4.3]        40.5 [39.5-44.1]        22.1 [22.1-22.3]         6.4x         0.2x         1.8x  (latency-bound)
   4096x4096    2137.1 [2132.6-2143.9]        58.5 [58.4-58.6]        84.2 [83.6-85.6]        72.1 [71.5-72.8]        29.6x         0.8x         1.2x
```

## rms_norm_rope (float16)

candidate: hipbridge.kernels.fused.rms_norm_rope (Triton, fused)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)         2 launches (us)          candidate (us)  vs original vs tuned HIPvs 2 launches
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       151.7 [151.6-151.7]           2.7 [2.6-2.7]        41.9 [40.2-42.2]        22.1 [22.0-22.2]         6.9x         0.1x         1.9x  (latency-bound)
   1024x1024       170.6 [170.6-170.6]           4.1 [4.1-4.1]        39.9 [39.6-40.3]        22.3 [22.1-24.9]         7.6x         0.2x         1.8x  (latency-bound)
   4096x4096    1167.7 [1165.8-1168.2]        29.5 [29.3-29.6]        47.7 [47.4-47.8]        41.1 [41.0-42.2]        28.4x         0.7x         1.2x
```

## rms_norm_rope (bfloat16)

candidate: hipbridge.kernels.fused.rms_norm_rope (Triton, fused)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)         2 launches (us)          candidate (us)  vs original vs tuned HIPvs 2 launches
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       183.8 [183.7-183.9]           2.7 [2.7-2.7]        49.7 [49.5-53.8]        27.6 [27.5-27.7]         6.7x         0.1x         1.8x  (latency-bound)
   1024x1024       202.7 [202.6-203.3]           4.2 [4.2-4.2]        40.8 [39.9-44.3]        22.1 [22.1-22.2]         9.2x         0.2x         1.8x  (latency-bound)
   4096x4096    1355.6 [1355.3-1356.9]        29.9 [29.9-30.1]        48.5 [47.2-48.7]        41.6 [41.1-42.9]        32.6x         0.7x         1.2x
```

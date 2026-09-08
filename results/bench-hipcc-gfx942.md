# hipbridge benchmark

| field | value |
|---|---|
| generated | 2026-09-08 05:06 UTC |
| hipbridge | 0.1.0.dev0 at commit `5bcfd99` |
| host | 2 (Linux x86_64) |
| device | AMD Radeon Graphics |
| toolchain | hipcc, HIP version: 7.14.60850-0000000 |
| arch | gfx942 |
| torch | 2.9.1+rocm6.4 |
| measured code | `1b2feaa6b6416e9c` |

## row_softmax (float32)

candidate: hipbridge.kernels.softmax (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       323.8 [323.6-324.0]           4.1 [4.1-4.1]           5.0 [4.8-5.4]        16.0 [15.9-16.2]        20.3x         0.3x         0.3x  (latency-bound)
   1024x1024       388.1 [387.6-388.5]           6.3 [6.3-6.3]           5.3 [5.3-5.3]        16.2 [15.9-16.3]        23.9x         0.4x         0.3x  (latency-bound)
   4096x4096    2427.9 [2427.5-2429.5]        52.2 [51.8-52.5]        40.5 [40.0-41.3]        31.6 [31.5-31.7]        77.0x         1.7x         1.3x
```

## row_softmax (float16)

candidate: hipbridge.kernels.softmax (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       308.2 [308.0-308.3]           4.0 [3.9-4.0]           5.2 [5.2-5.4]        16.3 [15.9-17.0]        18.9x         0.2x         0.3x  (latency-bound)
   1024x1024       341.4 [341.4-341.7]           6.4 [6.4-6.4]           6.7 [6.7-6.7]        16.1 [16.1-16.1]        21.2x         0.4x         0.4x  (latency-bound)
   4096x4096    2743.3 [2742.4-2744.8]        45.4 [44.7-45.5]        24.8 [23.8-25.6]        23.9 [23.8-23.9]       114.8x         1.9x         1.0x
```

## row_softmax (bfloat16)

candidate: hipbridge.kernels.softmax (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       338.1 [337.9-338.3]           4.0 [4.0-4.0]           5.6 [5.5-5.6]        16.0 [15.9-16.2]        21.2x         0.3x         0.4x  (latency-bound)
   1024x1024       372.3 [372.3-372.5]           6.6 [6.5-6.6]           7.2 [7.2-7.2]        16.1 [16.0-16.2]        23.2x         0.4x         0.4x  (latency-bound)
   4096x4096    3224.0 [3223.5-3224.0]        46.6 [45.2-47.3]        26.5 [25.3-28.1]        25.5 [25.3-26.0]       126.4x         1.8x         1.0x
```

## layer_norm (float32)

candidate: hipbridge.kernels.norm.layer_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       204.8 [204.8-205.0]           4.0 [3.9-4.0]           8.1 [8.0-8.3]        16.4 [16.4-16.7]        12.5x         0.2x         0.5x  (latency-bound)
   1024x1024       233.4 [233.4-233.6]           5.4 [5.4-5.4]           8.2 [8.2-8.4]        16.6 [16.4-16.7]        14.1x         0.3x         0.5x  (latency-bound)
   4096x4096    1737.6 [1735.5-1754.5]        46.0 [45.9-46.3]        44.6 [44.5-44.8]        30.9 [30.7-31.0]        56.2x         1.5x         1.4x
```

## layer_norm (float16)

candidate: hipbridge.kernels.norm.layer_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       272.0 [271.8-272.0]           3.9 [3.9-4.0]           8.3 [8.2-8.5]        16.1 [16.1-16.1]        16.9x         0.2x         0.5x  (latency-bound)
   1024x1024       297.6 [297.5-297.8]           5.6 [5.5-5.6]           8.2 [8.2-8.4]        16.4 [16.2-17.5]        18.2x         0.3x         0.5x  (latency-bound)
   4096x4096    1366.6 [1364.9-1367.2]        30.9 [30.5-31.0]        29.1 [27.6-29.6]        17.5 [17.5-17.6]        78.0x         1.8x         1.7x
```

## layer_norm (bfloat16)

candidate: hipbridge.kernels.norm.layer_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       291.7 [291.2-291.7]           4.0 [3.9-4.0]           8.2 [8.1-8.4]        16.2 [16.1-16.3]        18.0x         0.2x         0.5x  (latency-bound)
   1024x1024       317.3 [317.0-317.3]           5.6 [5.6-5.6]           8.5 [8.3-8.6]        16.7 [16.3-16.8]        19.1x         0.3x         0.5x  (latency-bound)
   4096x4096    1485.2 [1484.1-1485.2]        31.6 [31.4-32.2]        28.5 [28.3-30.5]        18.5 [18.1-18.6]        80.3x         1.7x         1.5x
```

## layer_norm_affine (float32)

candidate: hipbridge.kernels.norm.layer_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       208.9 [208.9-209.0]           4.1 [4.0-4.1]           8.2 [8.1-8.3]        19.6 [19.5-19.7]        10.6x         0.2x         0.4x  (latency-bound)
   1024x1024       241.0 [240.8-241.2]           5.5 [5.5-5.5]           8.4 [8.3-8.5]        19.3 [19.3-19.7]        12.5x         0.3x         0.4x  (latency-bound)
   4096x4096    1812.4 [1795.9-1821.1]        46.5 [46.2-46.7]        45.0 [44.9-45.3]        32.8 [32.8-33.2]        55.2x         1.4x         1.4x
```

## layer_norm_affine (float16)

candidate: hipbridge.kernels.norm.layer_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       276.1 [276.0-276.2]           4.0 [4.0-4.0]           8.4 [8.3-8.5]        19.4 [19.3-19.5]        14.2x         0.2x         0.4x  (latency-bound)
   1024x1024       306.0 [305.8-306.5]           5.7 [5.7-5.7]           8.5 [8.5-8.7]        19.3 [19.3-19.7]        15.9x         0.3x         0.4x  (latency-bound)
   4096x4096    1425.7 [1424.1-1425.9]        33.5 [33.4-33.5]        31.5 [30.5-31.5]        19.6 [19.5-19.7]        72.8x         1.7x         1.6x
```

## layer_norm_affine (bfloat16)

candidate: hipbridge.kernels.norm.layer_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       301.4 [301.4-301.5]           4.0 [4.0-4.0]           8.3 [8.2-8.4]        19.5 [19.4-19.6]        15.5x         0.2x         0.4x  (latency-bound)
   1024x1024       332.7 [332.6-333.1]           5.8 [5.8-5.8]           8.6 [8.5-8.9]        19.8 [19.7-19.8]        16.8x         0.3x         0.4x  (latency-bound)
   4096x4096    1581.8 [1580.5-1582.2]        35.1 [34.9-36.2]        31.0 [30.9-32.2]        20.6 [20.5-21.0]        76.8x         1.7x         1.5x
```

## rms_norm (float32)

candidate: hipbridge.kernels.norm.rms_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       169.9 [169.8-170.0]           2.8 [2.8-2.8]           7.0 [7.0-7.0]        16.5 [16.4-16.5]        10.3x         0.2x         0.4x  (latency-bound)
   1024x1024       193.7 [193.5-193.8]           4.1 [4.0-4.1]           7.4 [7.3-7.6]        16.6 [16.6-16.7]        11.7x         0.2x         0.4x  (latency-bound)
   4096x4096    1457.5 [1454.5-1471.4]        41.2 [41.2-41.3]        42.7 [42.7-42.9]        31.1 [31.1-31.2]        46.8x         1.3x         1.4x
```

## rms_norm (float16)

candidate: hipbridge.kernels.norm.rms_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       202.6 [202.4-202.8]           2.8 [2.8-2.8]           7.1 [7.0-7.2]        16.3 [16.2-16.6]        12.5x         0.2x         0.4x  (latency-bound)
   1024x1024       227.9 [227.8-228.0]           4.1 [4.1-4.1]           7.4 [7.4-7.7]        16.6 [16.4-16.7]        13.7x         0.2x         0.4x  (latency-bound)
   4096x4096    1057.3 [1056.3-1057.6]        22.7 [22.6-23.1]        20.1 [19.5-20.2]        17.1 [16.9-17.2]        61.9x         1.3x         1.2x
```

## rms_norm (bfloat16)

candidate: hipbridge.kernels.norm.rms_norm (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       220.2 [220.0-220.3]           2.9 [2.9-2.9]           7.1 [7.0-7.2]        16.5 [16.4-16.9]        13.3x         0.2x         0.4x  (latency-bound)
   1024x1024       247.4 [247.3-247.6]           4.2 [4.2-4.2]           7.2 [7.0-7.4]        16.6 [16.5-16.6]        14.9x         0.3x         0.4x  (latency-bound)
   4096x4096    1145.1 [1144.8-1145.6]        24.7 [24.4-24.8]        21.5 [20.5-21.7]        17.7 [17.7-17.7]        64.8x         1.4x         1.2x
```

## rms_norm_affine (float32)

candidate: hipbridge.kernels.norm.rms_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       173.7 [173.7-173.7]           2.8 [2.8-2.8]           7.2 [7.0-7.2]        18.1 [18.0-18.2]         9.6x         0.2x         0.4x  (latency-bound)
   1024x1024       200.4 [200.2-200.5]           4.0 [4.0-4.1]           7.3 [7.2-7.6]        18.5 [18.4-18.6]        10.9x         0.2x         0.4x  (latency-bound)
   4096x4096    1580.5 [1575.6-1585.6]        41.6 [41.2-41.6]        43.4 [43.2-43.5]        31.9 [31.6-32.0]        49.5x         1.3x         1.4x
```

## rms_norm_affine (float16)

candidate: hipbridge.kernels.norm.rms_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       214.4 [214.3-214.4]           2.9 [2.9-2.9]           7.9 [7.8-8.0]        20.3 [20.2-20.5]        10.6x         0.1x         0.4x  (latency-bound)
   1024x1024       240.0 [239.6-240.8]           4.2 [4.2-4.2]           7.4 [7.2-7.5]        18.1 [18.1-18.4]        13.2x         0.2x         0.4x  (latency-bound)
   4096x4096    1129.3 [1129.0-1129.5]        24.9 [24.4-25.2]        21.0 [20.7-21.9]        18.0 [18.0-18.2]        62.6x         1.4x         1.2x
```

## rms_norm_affine (bfloat16)

candidate: hipbridge.kernels.norm.rms_norm_affine (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       234.0 [233.9-234.0]           2.9 [2.9-2.9]           7.1 [7.1-7.2]        17.8 [17.8-18.1]        13.1x         0.2x         0.4x  (latency-bound)
   1024x1024       265.4 [265.3-265.5]           4.2 [4.2-4.2]           7.5 [7.4-7.6]        18.5 [18.4-18.6]        14.3x         0.2x         0.4x  (latency-bound)
   4096x4096    1246.4 [1245.9-1246.7]        26.2 [26.2-26.6]        23.1 [21.7-23.3]        18.3 [18.2-18.3]        68.3x         1.4x         1.3x
```

## rope (float32)

candidate: hipbridge.kernels.rope (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024          80.4 [80.4-80.4]           2.2 [2.1-2.3]        47.7 [46.2-48.6]        19.8 [19.5-19.8]         4.1x         0.1x         2.4x  (latency-bound)
   1024x1024       102.1 [101.8-102.1]           2.8 [2.8-2.8]        46.4 [46.2-47.5]        19.8 [19.7-19.9]         5.2x         0.1x         2.3x  (latency-bound)
   4096x4096    1373.8 [1373.1-1374.7]        45.5 [45.5-45.7]     237.1 [237.1-238.3]        50.6 [50.5-50.7]        27.2x         0.9x         4.7x
```

## rope (float16)

candidate: hipbridge.kernels.rope (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024          79.5 [79.5-79.6]           2.3 [2.2-2.3]        51.8 [50.3-53.3]        19.8 [19.4-22.4]         4.0x         0.1x         2.6x  (latency-bound)
   1024x1024          97.1 [97.0-97.2]           2.7 [2.7-2.7]        46.7 [46.5-47.0]        19.8 [19.8-20.1]         4.9x         0.1x         2.4x  (latency-bound)
   4096x4096       737.1 [736.5-737.7]        23.4 [23.4-23.7]     138.6 [137.8-141.0]        26.6 [26.4-27.3]        27.7x         0.9x         5.2x
```

## rope (bfloat16)

candidate: hipbridge.kernels.rope (Triton, AMD-tuned)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       109.5 [109.5-109.6]           2.2 [2.2-2.3]        46.6 [46.1-47.5]        19.6 [19.5-19.8]         5.6x         0.1x         2.4x  (latency-bound)
   1024x1024       127.3 [127.3-127.4]           2.8 [2.8-2.8]        46.1 [45.9-46.9]        19.8 [19.6-19.8]         6.4x         0.1x         2.3x  (latency-bound)
   4096x4096       934.2 [933.9-935.3]        23.8 [23.6-23.8]     138.5 [138.4-141.8]        27.0 [26.5-27.1]        34.6x         0.9x         5.1x
```

## rms_norm_rope (float32)

candidate: hipbridge.kernels.fused.rms_norm_rope (Triton, fused)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)         2 launches (us)          candidate (us)  vs original vs tuned HIPvs 2 launches
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       120.7 [120.7-120.8]           2.7 [2.6-2.7]        39.9 [39.7-40.1]        22.4 [22.3-22.5]         5.4x         0.1x         1.8x  (latency-bound)
   1024x1024       140.8 [140.7-140.8]           4.3 [4.3-4.3]        39.4 [39.3-39.4]        22.3 [22.1-22.9]         6.3x         0.2x         1.8x  (latency-bound)
   4096x4096    2138.2 [2117.6-2139.2]        58.4 [58.4-58.6]        86.4 [85.8-87.9]        73.4 [73.2-74.8]        29.1x         0.8x         1.2x
```

## rms_norm_rope (float16)

candidate: hipbridge.kernels.fused.rms_norm_rope (Triton, fused)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)         2 launches (us)          candidate (us)  vs original vs tuned HIPvs 2 launches
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       151.6 [151.6-151.6]           2.6 [2.6-2.6]        39.3 [39.3-39.4]        22.4 [22.4-22.6]         6.8x         0.1x         1.8x  (latency-bound)
   1024x1024       172.3 [170.2-172.8]           4.1 [4.1-4.1]        39.1 [39.1-39.2]        22.2 [22.1-22.4]         7.8x         0.2x         1.8x  (latency-bound)
   4096x4096    1163.3 [1162.0-1163.7]        29.5 [29.4-30.2]        47.3 [46.6-47.4]        41.2 [41.0-42.6]        28.2x         0.7x         1.1x
```

## rms_norm_rope (bfloat16)

candidate: hipbridge.kernels.fused.rms_norm_rope (Triton, fused)
reps: 100, runs: 3, device: `cuda`

```
       shape             original (us)          tuned HIP (us)         2 launches (us)          candidate (us)  vs original vs tuned HIPvs 2 launches
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       183.9 [183.8-183.9]           2.7 [2.7-2.8]        39.5 [39.0-39.6]        22.5 [22.4-22.6]         8.2x         0.1x         1.8x  (latency-bound)
   1024x1024       203.3 [203.1-203.3]           4.2 [4.2-4.2]        39.4 [39.1-40.2]        22.4 [22.3-22.6]         9.1x         0.2x         1.8x  (latency-bound)
   4096x4096    1355.0 [1353.3-1358.4]        30.3 [30.0-30.4]        48.4 [47.4-48.4]        41.5 [41.4-43.0]        32.7x         0.7x         1.2x
```

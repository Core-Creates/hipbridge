# hipbridge benchmark

| field | value |
|---|---|
| generated | 2026-09-07 19:40 UTC |
| hipbridge | 0.1.0.dev0 at commit `89e542b` |
| host | 2 (Linux x86_64) |
| device | no device visible to torch |
| toolchain | hipcc, HIP version: 7.14.60850-0000000 |
| arch | gfx942 |
| torch | 2.14.0+cu130 |

## row_softmax

candidate: torch.softmax (Triton unavailable)
reps: 100, runs: 3, device: `cpu`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       326.5 [326.3-326.6]           4.0 [4.0-4.0]           1.6 [1.6-1.6]           1.5 [1.4-1.5]  NOT COMPARABLE: candidate ran on cpu, not the device
   1024x1024       382.5 [381.4-383.6]           5.6 [5.6-5.6]        43.0 [42.4-49.5]        44.9 [42.3-49.3]  NOT COMPARABLE: candidate ran on cpu, not the device
   4096x4096    2429.3 [2428.7-2429.9]        93.4 [93.0-93.4]  4401.5 [4400.0-4640.7]  4446.3 [4318.0-4472.2]  NOT COMPARABLE: candidate ran on cpu, not the device
```

## layer_norm

candidate: torch.nn.functional.layer_norm (Triton unavailable)
reps: 100, runs: 3, device: `cpu`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       205.0 [204.8-205.0]           4.0 [3.9-4.0]           3.3 [3.3-3.4]           3.3 [3.3-3.3]  NOT COMPARABLE: candidate ran on cpu, not the device
   1024x1024       233.8 [233.8-233.9]           5.0 [5.0-5.0]        70.4 [58.4-74.0]        58.2 [56.8-59.9]  NOT COMPARABLE: candidate ran on cpu, not the device
   4096x4096    1748.1 [1742.8-1749.8]        50.6 [50.4-50.7]  4453.7 [4441.3-4816.0]  4454.4 [4424.4-4764.0]  NOT COMPARABLE: candidate ran on cpu, not the device
```

## layer_norm_affine

candidate: torch layer_norm affine (Triton unavailable)
reps: 100, runs: 3, device: `cpu`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       209.1 [209.0-209.3]           3.9 [3.9-4.0]           3.0 [3.0-3.1]           2.9 [2.9-2.9]  NOT COMPARABLE: candidate ran on cpu, not the device
   1024x1024       241.3 [241.2-241.3]           5.1 [5.1-5.1]        29.4 [29.1-30.0]        30.1 [29.4-37.0]  NOT COMPARABLE: candidate ran on cpu, not the device
   4096x4096    1854.0 [1853.6-1859.8]        51.6 [51.2-51.8]  4276.3 [4273.1-4653.0]  4261.4 [4088.3-4271.1]  NOT COMPARABLE: candidate ran on cpu, not the device
```

## rms_norm

candidate: torch rms_norm (Triton unavailable)
reps: 100, runs: 3, device: `cpu`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       169.9 [169.6-170.0]           2.9 [2.9-2.9]           8.0 [7.8-8.5]           7.9 [7.8-8.0]  NOT COMPARABLE: candidate ran on cpu, not the device
   1024x1024       193.7 [193.6-193.7]           4.0 [4.0-4.1]        60.1 [58.7-66.3]        58.5 [57.8-59.1]  NOT COMPARABLE: candidate ran on cpu, not the device
   4096x4096    1438.7 [1438.2-1440.3]        41.2 [41.2-41.3] 9859.0 [9194.7-10115.2] 9982.5 [9722.2-10088.6]  NOT COMPARABLE: candidate ran on cpu, not the device
```

## rms_norm_affine

candidate: torch rms_norm affine (Triton unavailable)
reps: 100, runs: 3, device: `cpu`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024       173.8 [173.8-173.9]           2.9 [2.9-2.9]           9.5 [9.5-9.9]           8.6 [8.5-8.6]  NOT COMPARABLE: candidate ran on cpu, not the device
   1024x1024       200.5 [200.3-200.6]           4.0 [4.0-4.0]        80.7 [80.5-83.1]        78.7 [73.9-79.9]  NOT COMPARABLE: candidate ran on cpu, not the device
   4096x4096    1569.8 [1566.2-1578.0]        41.6 [41.3-41.6]14564.4 [14335.2-14675.4]14496.6 [14162.3-14531.3]  NOT COMPARABLE: candidate ran on cpu, not the device
```

## rope

candidate: torch rope (Triton unavailable)
reps: 100, runs: 3, device: `cpu`

```
       shape             original (us)          tuned HIP (us)              torch (us)          candidate (us)  vs original vs tuned HIP     vs torch
-----------------------------------------------------------------------------------------------------------------------------------------------------
      1x1024          80.4 [80.3-80.5]           2.2 [2.1-2.3]        16.4 [16.3-16.4]        16.3 [16.3-16.5]  NOT COMPARABLE: candidate ran on cpu, not the device
   1024x1024       101.7 [101.7-101.8]           2.8 [2.8-2.9]     260.4 [246.9-261.1]     263.8 [260.6-265.8]  NOT COMPARABLE: candidate ran on cpu, not the device
   4096x4096    1363.1 [1362.7-1364.2]        45.8 [45.3-45.9]20494.0 [20351.6-20591.8]20041.8 [19634.1-20256.8]  NOT COMPARABLE: candidate ran on cpu, not the device
```

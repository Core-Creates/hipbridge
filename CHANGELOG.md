# Changelog

## 0.1.0

First release. Pre-alpha, and the classifier says so: this is published to be
installable and looked at, not to be depended on.

### What it does

Recognizes a CUDA kernel, proposes a tuned AMD Triton implementation, and proves
the substitution numerically against a float64 oracle rather than against the
original's rounding. An unrecognized kernel returns UNKNOWN and a report; it
never falls through to a nearest match.

### What is covered

Seven row-wise kernels: `row_softmax`, `layer_norm`, `layer_norm_affine`,
`rms_norm`, `rms_norm_affine`, `rope`, `rms_norm_rope`. All seven verify at
93/93 cases and prove end to end on an MI300X (gfx942), in float32, float16 and
bfloat16.

### What it will not claim

Five of the seven beat a competently written HIP kernel at large shapes, by 1.3x
to 1.6x in float32. `rope` and `rms_norm_rope` lose at every shape and precision
measured, and `port` says so beside the proof. Below roughly 16M elements every
substitution is a 3x to 11x regression. Correctness and value are different
questions and this release answers both separately.

### Known limits

- Verified on one device: an MI300X VF partition, gfx942. `gfx90a` has never
  been run. The NVIDIA path has been exercised for softmax only, on one RTX 4060.
- `hipbridge verify` from an installed copy needs `--examples` pointed at a
  checkout; the `.cu` files are not packaged.
- No integration path beyond editing your own call sites. There is no
  torch.compile backend and no module swap.
- The kernels covered are row-wise normalisation and softmax, which are not
  where a transformer spends most of its time.

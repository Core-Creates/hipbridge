# hipbridge

Recognize CUDA kernels, substitute verified AMD implementations, prove it numerically.

**Status: pre-alpha.** Correctness has been verified on real hardware, on both
vendors: nvcc on an RTX 4060 via WSL2, and hipcc on an MI300X (gfx942).
Performance has been measured on the MI300X, 17.8x to 76.8x, with the
measurement conditions stated alongside it. See
[the status table](#status-of-what-has-actually-been-run) for exactly which
paths those are. Claims in this README are limited to what has actually been
run, never to what should follow from it.

## Design rules

1. **Never emit a kernel we cannot verify.** An unrecognized kernel returns
   `UNKNOWN` and a report. It does not fall through to a default pattern.
2. **The verifier gates the translator, not the other way round.**
   `hipbridge.verify` was built first and is usable entirely on its own.
3. **Core never imports an extra.** `frontend`, `recognize`, and `analysis` must
   not import `hipbridge.verify` or `hipbridge.kernels`. Enforced by
   `tests/test_layering.py`.

## Install

```bash
pip install hipbridge              # core: frontend + recognize + analysis (libclang only)
pip install hipbridge[kernels]     # + tuned AMD Triton implementations
pip install hipbridge[verify]      # + differential harness (torch)
pip install hipbridge[all]
```

Core installs anywhere. The `kernels` and `verify` extras pull Triton, which
publishes Linux wheels only; the dependency is marked so it degrades instead of
failing the install.

## Recognizing a kernel

```
$ hipbridge inspect examples/tree_reduce.cu
kernel:  tree_reduce
pattern: reduce_tree
confidence: certain
  - [reduce.tree] 1 shared buffer(s): float[256] (1024 B)
  - [reduce.tree] 2 __syncthreads() calls
  - [reduce.tree] loop stride is halved (log-depth tree)
  - [reduce.tree] AMD note: wavefront is 64 wide, tree needs 6 steps not 5
```

When nothing claims the kernel, it says so instead of guessing:

```
$ hipbridge inspect examples/tiled_transpose.cu
pattern: unknown
  - no recognizer claimed this kernel (4 tried)
  No substitution will be attempted. Translate this kernel by hand or extend the recognizers.
```

## Verifying an implementation

```python
import torch
from hipbridge import verify
from hipbridge.verify import shapes

harness = verify.Harness(
    candidate=my_triton_softmax,
    reference=verify.TorchReference(lambda x: torch.softmax(x, dim=-1)),
)
print(harness.run(shapes.row_wise()))
```

The sweep crosses shapes against input distributions chosen to break things
rather than confirm them. Shapes straddle the 64-wide AMD wavefront (63, 64, 65)
because that is where mask errors live. Distributions include large-magnitude
inputs specifically because a softmax missing its max-subtraction pass is
**bitwise correct on N(0,1)** and destroyed on realistic logits.

Four failure modes are detected explicitly:

| Mode | What it catches |
|---|---|
| `IDENTITY` | Output is a copy of the input, computation dropped |
| `unwritten` | Output buffer never stored to (sentinel-filled before launch) |
| `NONDETERMINISTIC` | Same input, different answer across runs. NaN-aware |
| numeric | Answers differ beyond the ULP budget |

Against a deliberately broken softmax set, 84 cases each:

```
PASS  correct softmax:          84/84 cases, worst ulp=0
FAIL  identity kernel:           0/84 cases
FAIL  dropped max-subtraction:  62/84 cases   <- passes on N(0,1), fails on large
FAIL  reduced the wrong axis:    7/84 cases
```

The 62/84 row is the point of the whole design. A harness that only tested
N(0,1) would have passed that kernel.

### References

`TorchReference` wraps a PyTorch oracle and always works. `NativeReference`
compiles the original `.cu` with `nvcc` (or hipified source with `hipcc`) and
runs it on a real device, which is the only reference that proves anything about
the kernel you were actually handed. Availability is detected, never assumed: a
missing toolchain yields an unavailable reference with a reason, not a crash.

## Verifying against real hardware

```bash
hipbridge verify --toolchain nvcc --wsl Ubuntu          # NVIDIA, via WSL2
hipbridge verify --toolchain hipcc --arch gfx942        # AMD MI300X
```

No device present is a **skip at exit 0**, not a failure, because this runs
mostly on machines without a GPU. Pass `--require` to make absence fatal.

`.github/workflows/amd-verify.yml` is **manual only** (`workflow_dispatch`).
MI300X time is metered at roughly $1.71 to $3.47 per GPU-hour, so it never
fires on push; the free eight-job matrix in `ci.yml` covers everything that
does not need a device. The AMD job carries a 30 minute hard timeout and a
shape-limit input, and three tests assert the cost model directly: the AMD
workflow must stay `workflow_dispatch` only, it must have a timeout, and no
job in `ci.yml` may request a self-hosted runner.

Run it with `runner: ubuntu-latest` for a free dry run that proves the
plumbing without spending anything. Switch to `self-hosted` once a runner is
registered on a rented box.

### On an AMD ROCm box

```bash
gh auth login && gh repo clone Core-Creates/hipbridge
cd hipbridge

bash scripts/smoke-hip.sh      # toolchain only: no Python, no pip, no PyTorch
bash scripts/bootstrap-amd.sh  # full harness (creates .venv; PEP 668 blocks system pip)
```

Run the smoke test first. It compiles and runs `row_softmax` on the device and
self-checks against a float64 CPU computation, so it separates "does hipcc
work" from "is the Python environment set up". Some AMD Developer Cloud images
ship hipcc and the driver but **no pip and no PyTorch**; the bootstrap script
handles that, the smoke test does not need it.

ROCm images ship a ROCm-built PyTorch plus `pytorch-triton-rocm`. **Do not
reinstall either from PyPI**, which would replace them with NVIDIA builds and
break the GPU you are paying for. The `verify` extra therefore declares `torch`
and nothing else; on such a box Triton arrives with the platform PyTorch, and
`hipbridge.kernels` picks it up from there. A test asserts the extra never
regrows a `triton` dependency.

### Status of what has actually been run

| Path | State |
|---|---|
| CPU, torch oracle | verified |
| NVIDIA, nvcc on RTX 4060 via WSL2 | verified, 56/56 cases |
| AMD, hipcc on MI300X (gfx942) | **verified**, 84/84 cases |
| AMD, timed on MI300X (gfx942) | **measured**, 17.8x to 76.8x |

## The result this project was built to get

On an AMD Instinct MI300X (gfx942, HIP 7.14, torch 2.9.1+rocm6.4), the tuned
Triton kernel checked against the original `row_softmax.cu` compiled with
`hipcc` and executed on device:

```
candidate: hipbridge.kernels.softmax (Triton, AMD-tuned)
PASS  row_softmax vs original on hipcc: 84/84 cases, worst ulp=593
      [accuracy vs original: better=6, equivalent=78, up to 297x closer to float64]
```

**593 ULP of divergence, and it passes.** That is the entire argument.

The Triton kernel reduces pairwise across a 64-wide wavefront; the original
accumulates serially, which grows rounding error as O(n) rather than O(log n).
So the two disagree enormously, and where the disagreement is large enough to
resolve, the Triton one is *closer to float64 truth*.

Judged the obvious way, "does the translation match the original within a ULP
budget", this run scores **0/84** and the better kernel is rejected. Judged
against a float64 oracle, it scores 84/84. Matching the original would have
meant reproducing its rounding error.

Read the verdict breakdown precisely, because it is a narrower claim than "more
accurate everywhere":

- On **78 of 84** cases both implementations land at the float32 noise floor.
  Neither is meaningfully closer to truth and the tool calls that equivalent.
- On **6** cases the original climbs off the floor and the Triton kernel does
  not, and there it is up to **297x** closer to float64. Those are the inputs
  the sweep exists to generate: sparse, mixed-sign, and large-magnitude.
- **Never worse.** Not once in 84.

So the honest statement is not that the substitution is always more accurate.
It is that the two are indistinguishable on ordinary inputs, and on the inputs
built to expose serial accumulation the substitution is far closer to truth and
never further from it.

The same effect was first measured on an RTX 4060, where the original kernel
was 1.4x to 59.7x less accurate than torch across every shape and distribution
tried. See `compare.arbitrate` and `tests/test_arbitration.py`.

First MI300X run, `scripts/smoke-hip.sh`, HIP 7.14, gfx942:

```
max |row sum - 1| : 3.325e-07
max abs error     : 1.307e-08  (vs float64 CPU)
PASS: hipcc built it, MI300X ran it, the numbers are right.
```

1.3e-08 is well inside float32 epsilon (1.19e-07). The sentinel fill confirmed
every output element was written, so this is not a kernel that silently skipped
its store.

**On the header skew, since the workaround is still in the code.** That box
first ran with Ubuntu's `libamdhip64-dev` 5.7.1 headers in `/usr/include/hip`
against a 7.14 compiler in `/opt/rocm/core-7.14`, which leaves
`__AMDGCN_WAVEFRONT_SIZE` undefined and forces the retry with
`-D__AMDGCN_WAVEFRONT_SIZE=64`. Installing `amdrocm-runtime-dev7.14` puts
matching 7.14.60850 headers under `/opt/rocm/include` and removing the 5.7
package clears the ambiguity, after which the compile line carries no `-D` at
all. Everything published here was re-run on matched headers.

Worth recording: the numbers did not move. The smoke test returned bit-identical
figures before and after, and the timings shifted less than run-to-run variance.
The workaround was harmless, which was a reasonable guess and is now a measured
fact rather than a hope. `scripts/install-hip-headers.sh` still leads with
`libamdhip64-dev`, which is the trap; on ROCm 7 the package you want is
`amdrocm-runtime-dev7.14`.

## Benchmarking against the original

Correctness does not justify a substitution on its own. If the tuned kernel is
not faster there is no reason to swap it in, so `bench` times both sides on the
same device, at the same shape, with one stated method.

```bash
hipbridge bench --toolchain hipcc --arch gfx942 --reps 100
hipbridge bench --toolchain nvcc --wsl Ubuntu --shapes 1x1024,4096x4096
```

Defaults are `--shapes 1x1024,64x1024,1024x1024,4096x4096` and `--reps 100`.
`--examples`, `--wsl`, and `--require` behave as they do for `verify`.

Method, stated in `verify/bench.py` because a benchmark without one is an
anecdote:

- The original is timed **inside the generated driver** with device events, so
  file I/O and host copies are excluded. Those otherwise dominate and make every
  kernel look identical.
- The candidate is timed with device events too, never wall clock.
- Both sides get warmup iterations before any timing, to pay JIT and cache costs
  once rather than charge them to the first measurement.
- The figure reported is the per-iteration mean over `reps` back-to-back
  launches.
- **A shape is timed only after it verified correct at that shape.** Shapes that
  did not verify are still printed, tagged `(UNVERIFIED at this shape)`. A fast
  wrong kernel is not a result.

### It refuses to report a ratio it cannot stand behind

The reference always runs on device. If torch put the candidate on the host, a
ratio between them compares a CPU implementation against a GPU one and means
nothing, so no ratio is printed:

```
     64x1024  original     575.7 us (device)   candidate    2375.0 us (cpu)   NOT COMPARABLE: candidate did not run on the device
```

That guard exists because the first run of this command, on a Windows box with
a CPU-only torch wheel, cheerfully reported a **3.6x speedup** for exactly that
mismatch. Footnoting it would not have been enough; the number would still have
been quoted. When both sides are genuinely on device the line ends in a speedup
ratio instead.

A CPU-only torch install is announced up front rather than left in the output
for a reader to catch:

```
WARNING: no GPU visible to torch, so the candidate runs on the host
         while the original runs on device. Ratios are suppressed as
         NOT COMPARABLE. On Windows this is normally a CPU-only torch
         wheel; Triton has no Windows build either.
```

### Measured on an MI300X

```
row_softmax on hipcc, device=cuda, reps=100
  candidate: hipbridge.kernels.softmax (Triton, AMD-tuned)
  original : row_softmax.cu compiled with hipcc

      1x1024  original     323.7 us   candidate      18.1 us     17.8x
     64x1024  original     334.9 us   candidate      17.7 us     18.9x
   1024x1024  original     388.3 us   candidate      18.2 us     21.3x
   4096x4096  original    2434.3 us   candidate      31.7 us     76.8x
```

Conditions, because a ratio without them is not a measurement:

- AMD Instinct MI300X **VF** (a virtualized partition, not a whole card),
  gfx942, HIP 7.14.60850 with matched headers, torch 2.9.1+rocm6.4
- 100 reps per shape after warmup, device events both sides
- every shape verified correct at that shape before it was timed
- four independent runs, two of them by a different operator: the 4096x4096
  figure landed at 76.8x, 76.8x, 77.0x and 77.3x, and the small shapes within
  about 5%

**Read the comparison honestly. Much of this gap is the original's launch
configuration, not the language it is written in.** `row_softmax.cu` launches
with `block=(1,1,1)`, one thread per block, so it uses 1/64th of a wavefront and
leaves a 304-CU device essentially idle. A competently launched HIP kernel would
close a large part of the distance. What the table measures is the substitution
as a whole, a naive original replaced by a tuned implementation, which is the
transaction hipbridge actually offers.

The shape of the curve is more informative than any single ratio. The candidate
is flat at roughly 18 us from 1x1024 through 1024x1024, so at those sizes it is
launch-latency bound rather than compute bound, and only starts doing real work
at 4096x4096. The original is nearly flat too, 324 us to 388 us across a
1000-fold increase in data, which is the signature of serialization rather than
memory traffic: it is not moving bytes, it is waiting on one thread.

The prior expectation, recorded here so it can be checked rather than quietly
revised afterwards: the original `row_softmax.cu` launches with `block=(1,1,1)`,
one thread per block, which uses 1/64th of each wavefront on a 304-CU MI300X,
while the Triton kernel uses the full 64-wide wavefront with a tree reduction.
The gap should be large and should widen with row count. That is a prediction,
not a result, and the whole point of the command is that it can disprove it.

## Layout and the future split

`src/hipbridge/verify/` and `src/hipbridge/kernels/` are kept import-clean so
each can be promoted to its own distribution without a refactor. Promote on
evidence (independent users, independent issues), not on a hunch.

## License

Apache-2.0.

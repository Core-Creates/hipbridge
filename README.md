# hipbridge

Recognize CUDA kernels, substitute verified AMD implementations, prove it numerically.

**Status: pre-alpha.** Correctness has been verified on real hardware, on both
vendors: nvcc on an RTX 4060 via WSL2, and hipcc on an MI300X (gfx942).
Performance has been measured on the MI300X against three baselines: 17x to
76x against the naive original, **2.9x against a competently written HIP
kernel** at large shapes, and a 3x to 5x regression below roughly 16M elements.
See
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

## Porting a kernel, in one command

`inspect` recognizes and `verify` proves. `port` is both, which is the only one
of the three that answers the question a user actually has.

```
$ hipbridge port examples/row_softmax.cu --arch gfx942
kernel:  row_softmax
pattern: reduce_serial
confidence: likely

proposing: hipbridge.kernels.softmax (Triton, AMD-tuned)
  - calls expf, so it is not a plain sum
  - takes a maximum, the usual stability pass
  - divides by an accumulated total
  - one input and one output pointer (row_softmax)
  - two scalar arguments (rows, cols), read as rows/cols

proving against examples/row_softmax.cu compiled with hipcc
  launch: grid one block per row, block=(1, 1, 1)
PASS  row_softmax vs hipbridge.kernels.softmax: 84/84 cases, worst ulp=593
      [accuracy vs original: better=6, equivalent=78, up to 297x closer to float64]

SUBSTITUTION PROVED. Use it like this:

    from hipbridge.kernels.softmax import softmax_rowwise

    out = softmax_rowwise(x)   # replaces row_softmax
```

**A pattern is not a licence to substitute.** `row_softmax.cu` is recognized as
`reduce_serial`, and so is a kernel that sums a row, and so is one that takes a
product. Substituting a softmax into either would corrupt data silently, so a
proposal needs corroborating evidence in the source and then, crucially, a
numeric proof. Recognition proposes; the float64 oracle decides.

The proof compiles **your** kernel, not the copy shipped in `examples/`. Proving
a substitution against our own original would prove nothing about yours.

Exit codes are the interface:

| Code | Meaning |
|---|---|
| 0 | substituted and proved on device |
| 3 | nothing to propose; the kernel is unrecognized, or its pattern is not enough |
| 4 | proposed, but the proof failed, or the original itself is not sane |
| 5 | no toolchain or device to prove it on, with `--require` |

Refusals are the common case and are meant to be. Of the five kernels in
`examples/`, three are refused: `tiled_transpose.cu` because nothing recognizes
it, `tree_reduce.cu` and `saxpy.cu` because sharing a pattern with softmax is
not evidence of being softmax.

### It checks the original before trusting it

Oracle mode passes the candidate when it is closer to the truth than the
original is, which has a hole: if the original is garbage, the candidate is
trivially closer and the run reports a proof. That is not theoretical. An early
version of `port` reused the suite's launch geometry, so `row_softmax_tuned.cu`
ran at `block=(1,1,1)`, read uninitialised shared memory in its tree reduction,
and the harness scored the candidate **3.9e75x "better"** than the wreckage and
declared the substitution proved.

Two fixes, both kept: launch geometry is now inferred from the kernel itself
(no `threadIdx` means one thread per row; a shared buffer means a block as wide
as the buffer; `--block` overrides), and the original is checked against the
oracle before any comparison with it is believed. A reference that cannot
reproduce its own maths is not a baseline, and the run stops with exit 4.

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

bash scripts/smoke-hip.sh          # toolchain only: no Python, no pip, no PyTorch
bash scripts/install-hip-headers.sh  # if the smoke test cannot find hip/hip_runtime.h
bash scripts/bootstrap-amd.sh      # full harness (creates .venv; PEP 668 blocks system pip)
```

`install-hip-headers.sh` reads `hipcc --version` and installs the matching
`amdrocm-runtime-dev<version>`, then refuses to report success until a header
tree's `HIP_VERSION_MAJOR.MINOR` actually equals the compiler's. Run it with
`PURGE_STALE=1` to also remove a mismatched tree once a matching one exists.
**Do not install `libamdhip64-dev` by hand on Ubuntu 24.04**: it resolves to HIP
5.7.1 from `noble/universe`, installs happily beside a ROCm 7 compiler, and the
only symptom is one undefined-macro error that is trivially worked around. Both
scripts now check header and compiler versions against each other and say so.

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
| AMD, timed on MI300X (gfx942) | **measured**, 2.9x vs tuned HIP at scale, slower below it |

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
fact rather than a hope. `scripts/install-hip-headers.sh` used to lead with
`libamdhip64-dev`, which is what installed the skew in the first place; it now
derives the package from the compiler version and verifies the result. Rebuilt
the fault on the same box to check: with only the 5.7 headers present, one run
of the script installs `amdrocm-runtime-dev7.14`, removes the stale tree under
`PURGE_STALE=1`, and the smoke test, verify and bench all pass afterwards.

## Benchmarking against baselines that can win

Correctness does not justify a substitution on its own. If the tuned kernel is
not faster there is no reason to swap it in, so `bench` times every side on the
same device, at the same shape, with one stated method.

```bash
hipbridge bench --toolchain hipcc --arch gfx942 --reps 100 --runs 5
hipbridge bench --toolchain nvcc --wsl Ubuntu --shapes 1x1024,4096x4096
```

Defaults are `--shapes 1x1024,64x1024,1024x1024,4096x4096`, `--reps 100` and
`--runs 5`. `--examples`, `--wsl`, and `--require` behave as they do for
`verify`.

**Three baselines, not one**, because a benchmark is only as honest as the
thing it beats:

| Baseline | What it answers |
|---|---|
| `original` | what does replacing this exact kernel buy me |
| `tuned HIP` | would a competent engineer have done as well by hand |
| `torch` | why not just call the library |

`row_softmax.cu` launches one thread per block, so beating it is not evidence of
anything. `row_softmax_tuned.cu` is the same maths written properly, one block
per row with tree reductions in shared memory, and it is what the claim should
be measured against. `torch.softmax` is what a user has instead of any of this.

Method, stated in `verify/bench.py` because a benchmark without one is an
anecdote:

- Native kernels are timed **inside the generated driver** with device events,
  so file I/O and host copies are excluded. Those otherwise dominate and make
  every kernel look identical.
- The candidate and the torch baseline are timed with device events too, never
  wall clock, and through the same Python path so the two carry the same
  dispatch cost.
- Every side gets warmup iterations before any timing, to pay JIT and cache
  costs once rather than charge them to the first measurement.
- One run is the mean over `reps` back-to-back launches. The table reports the
  **median across `runs` of those, with min and max beside it**, because a lone
  mean hid a 15% spread on small shapes.
- Shapes where the candidate never approaches the throughput it reaches at
  larger sizes are labelled **`latency-bound`**. A ratio there describes
  dispatch overhead, not the kernel.
- **A shape is timed only after it verified correct at that shape.** Shapes that
  did not verify are still printed, tagged `(UNVERIFIED at this shape)`. A fast
  wrong kernel is not a result. Additional baselines are arbitrated against the
  float64 oracle before being timed, so a fast baseline that computes the wrong
  thing cannot flatter the candidate.

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
row_softmax on hipcc, device=cuda, reps=100, runs=5
  candidate: hipbridge.kernels.softmax (Triton, AMD-tuned)
  original : row_softmax.cu compiled with hipcc
  tuned HIP: row_softmax_tuned.cu compiled with hipcc
  torch    : library call, timed like the candidate

       shape        original (us)     tuned HIP (us)         torch (us)     candidate (us)  vs original  vs tuned HIP  vs torch
      1x1024   326.1 [325.9-326.2]      3.9 [3.9-4.0]      5.5 [5.4-5.5]   17.1 [16.9-17.3]       19.1x         0.2x      0.3x  (latency-bound)
     64x1024   334.4 [334.4-334.4]      4.1 [4.1-4.2]      5.9 [5.6-5.9]   19.7 [19.5-20.4]       17.0x         0.2x      0.3x  (latency-bound)
   1024x1024   363.6 [363.4-364.0]      5.6 [5.6-5.6]      5.2 [5.2-5.5]   17.0 [15.4-18.2]       21.4x         0.3x      0.3x  (latency-bound)
   4096x4096  2430.4 [2429.3-2432.3]   93.5 [92.4-94.0]   40.3 [40.2-48.3]  31.8 [31.5-32.0]      76.4x         2.9x      1.3x
   8192x4096  3431.6 [3428.2-3433.2]  180.0 [178.6-180.4] 73.2 [72.0-80.0]  61.2 [60.4-61.3]      56.1x         2.9x      1.2x
```

Median across 5 runs of 100 reps, min and max beside it.

**The 76x is the least interesting number here, and it is close to meaningless.**
It is measured against `row_softmax.cu`, which launches one thread per block and
leaves a 304-CU device idle. Any competent kernel beats it. That is why the
table carries two baselines that can actually win.

Read the last two columns instead:

| Regime | vs a competent HIP kernel | vs `torch.softmax` | Verdict |
|---|---|---|---|
| Rows up to ~1M elements | **0.2x to 0.3x** | **0.3x** | the substitution is a **regression** |
| 16M elements and up | **2.9x** | **1.2x to 1.3x** | the substitution is worth making |

So the honest claim is not "77x faster". It is: **at large shapes the tuned
Triton kernel beats a competently written HIP kernel by 2.9x and torch by about
1.25x, and below roughly 16M elements it loses to both by 3x to 5x.**

The crossover is dispatch cost, and it is measurable rather than assumed. On
this box, an in-place torch op that does no work at all costs **4.9 us** to
launch from Python, `torch.softmax` on a 1x1024 row costs **5.5 us**, and the
Triton candidate costs **17 us**. About 12 us of that is Triton's own launch
path, and it is fixed, so it dominates until the kernel has real work to do.
Below the crossover the ratio describes dispatch overhead, not the kernel, which
is what the `latency-bound` label marks.

One asymmetry worth stating: the native baselines are timed inside the generated
C++ driver with device events, so they never pay Python dispatch, while the
candidate and torch are timed through Python and do. That flatters the native
side by roughly 5 us. It does not change any conclusion here, because the gaps
at small shapes are 12 us and more, and at large shapes the candidate wins
anyway.

`row_softmax_tuned.cu` is not a straw baseline either, and it is not optimal:
one block per row, 256 threads, two shared-memory tree reductions, three passes
over memory. torch beats it 2.3x at 4096x4096 because a fused implementation
moves less data. A better hand-written kernel would narrow the 2.9x, and if
someone writes one, that is a result worth having rather than an embarrassment.

Conditions, because a ratio without them is not a measurement:

- AMD Instinct MI300X **VF** (a virtualized partition, not a whole card),
  gfx942, HIP 7.14.60850 with matched headers, torch 2.9.1+rocm6.4
- every shape verified correct at that shape before it was timed, and each
  additional baseline arbitrated against the float64 oracle before being timed,
  so a fast baseline computing the wrong thing cannot flatter the candidate
- the naive original's own curve is the tell: 326 us to 3432 us across a
  32000-fold increase in data. It is not moving bytes, it is waiting on one
  thread.

The prediction this table was built to test was "the gap should be large and
should widen with row count". Against the naive original it held. Against a
competent kernel it was **wrong below 16M elements**, in the direction that
matters, and finding that out is the entire reason the second baseline exists.

## Layout and the future split


`src/hipbridge/verify/` and `src/hipbridge/kernels/` are kept import-clean so
each can be promoted to its own distribution without a refactor. Promote on
evidence (independent users, independent issues), not on a hunch.

## License

Apache-2.0.

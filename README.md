# hipbridge

Recognize CUDA kernels, substitute verified AMD implementations, prove it numerically.

**Status: pre-alpha.** Six kernels are covered, spanning the row-wise part of a
transformer's inference path, and all six verify on an MI300X at 84/84 cases
each. Correctness has been checked on both vendors: nvcc on an RTX 4060 via
WSL2, and hipcc on an MI300X (gfx942).

Performance is measured by CI on a self-hosted MI300X across all six kernels
and three precisions. At large shapes the substitutions beat a **competently
written HIP kernel by 1.3x to 1.7x** and torch by 1.3x to 1.4x, except RoPE
which loses to hand-written HIP; below roughly 16M elements every substitution
is a 3x to 5x regression. See
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

## What it covers

Six kernels, which is the row-wise part of a transformer's inference path. Each
ships a naive original, a competently written HIP baseline to benchmark
against, a Triton implementation, and a float64 oracle.

| Kernel | Extra operands | Substitute |
|---|---|---|
| `row_softmax` | none | `kernels.softmax.softmax_rowwise` |
| `layer_norm` | none | `kernels.norm.layer_norm_rowwise` |
| `layer_norm_affine` | `gamma`, `beta` | `kernels.norm.layer_norm_affine_rowwise` |
| `rms_norm` | none | `kernels.norm.rms_norm_rowwise` |
| `rms_norm_affine` | `gamma` | `kernels.norm.rms_norm_affine_rowwise` |
| `rope` | `cos_tab`, `sin_tab` | `kernels.rope.rope_rowwise` |

All six verify on an MI300X, 84/84 cases each. Two things about the list are
worth saying plainly rather than leaving to be discovered.

**The affine forms take learned weights, and that took a harness change.** A
kernel scaling by gamma cannot be proved by a harness that hands it one tensor,
and proving the normalisation alone while substituting something that also
multiplies by weights would prove one program and ship another. Suites now
declare their extra operands, which are generated per case, swept with the
shape, and handed to the candidate, the reference and the oracle alike.

**Operands are matched by name, not only by position.** The generated driver
binds input buffers in declaration order, so a LayerNorm written
`(in, beta, gamma, out, rows, cols)` would receive gamma where it expects beta.
It compiles, it runs, it returns finite numbers, and it is wrong on every row.
A recognised name in the wrong slot is refused and explained; an unrecognised
name is allowed through, because names are a weaker signal than the numeric
proof and refusing every unfamiliar spelling would reject correct kernels for
their vocabulary. `gamma` also answers to weight, scale, g and w.

**RoPE is not a reduction**, and it did not fit the recognizer. `reduce.serial`
requires an accumulation across the row and `elementwise.flat` requires no loop
at all, so RoPE was `UNKNOWN`, correctly. Rather than widen either rule until it
swallowed something it does not describe, there is a `row_map` pattern for a
thread that walks a row and accumulates nothing, claimed on that absence, which
is exactly what makes a row's outputs independent.

The accuracy argument that carries softmax and the norms does not carry RoPE,
and this README will not pretend otherwise: there is nothing to reduce, so
there is no serial accumulation to beat. On the MI300X the substitution is
bitwise identical to the original, `worst ulp=0`, equivalent on all 84 cases.
The case for substituting it is throughput, not numerics.

Neither the affine norms nor the plain ones fuse anything, and `rope` assumes an
even head dimension, since pairing channel 2i with 2i+1 has no meaning
otherwise. Odd widths are dropped from its sweep rather than silently truncated
and counted as passing.

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
  - no recognizer claimed this kernel (5 tried)
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

Refusals are meant to happen. Of the 15 kernels in `examples/`, three are
refused: `tiled_transpose.cu` because nothing recognizes it, and `tree_reduce.cu`
and `saxpy.cu` because sharing a pattern with a substitutable kernel is not
evidence of being one. `tree_reduce.cu` is a `reduce_tree` exactly as
`row_softmax_tuned.cu` is, and it sums where the other normalises.

The other twelve are the six covered kernels, each in a naive and a tuned
form, and the pair reaching the same substitute is the point: written badly or
written well, it is still the same maths.

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
| NVIDIA, nvcc on RTX 4060 via WSL2 | verified, 56/56 cases, softmax only |
| AMD, hipcc on MI300X (gfx942) | **verified**, all six suites, 84/84 cases each |
| AMD, kernels taking learned weights | **verified**, `layer_norm_affine` and `rms_norm_affine` |
| AMD, timed on MI300X (gfx942) | **measured**, softmax and the plain norms |
| AMD, timed with weights | not yet measured |

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

### The same run, across all six kernels

```
PASS  row_softmax        84/84  ulp=593         (max abs 1.199e-07)  better=6,  equivalent=78, up to 297x closer
PASS  layer_norm         84/84  ulp=1776828265  (max abs 3.910e-05)  better=5,  equivalent=79, up to  79x closer
PASS  layer_norm_affine  84/84  ulp=332160      (max abs 6.714e-04)  better=5,  equivalent=79, up to  86x closer
PASS  rms_norm           84/84  ulp=10          (max abs 1.907e-06)  equivalent=84,            up to  15x closer
PASS  rms_norm_affine    84/84  ulp=10          (max abs 3.662e-04)  equivalent=84,            up to  11x closer
PASS  rope               84/84  ulp=0           (max abs 0.000e+00)  equivalent=84
```

Three of those lines need reading carefully, and the absolute error beside each
ULP count is why it is printed.

**`layer_norm` diverges by 1.78 billion ULP and is fine.** Its outputs are
centred, so they sit near zero, and ULP distance explodes there: +1e-9 and -1e-9
are a hair apart in magnitude and astronomically far apart on the integer line.
The absolute error is 3.9e-05. ULP is the right scale-free metric for softmax,
whose outputs are positive and O(1), and the wrong one for anything crossing
zero.

**`rope` is bitwise identical to the original**, `ulp=0`, equivalent on every
case. There is no reduction in RoPE, so there is no serial accumulation to beat,
and the accuracy argument simply does not apply. Substituting it is a throughput
decision.

**The two `equivalent=84` rows used to read `better=12`.** `arbitrate` scored a
tie as a win, because a tie gives a ratio of exactly 1.0 and the test was
`ratio <= 1.0`. Tightening it to `< 1.0` removed 24 false wins across the sweep
and left the 16 real ones untouched, which is the useful part: the softmax and
LayerNorm advantages were not artifacts.

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

Every row below was produced by CI on a self-hosted MI300X, not by a person at a
terminal, and the reports are committed under `results/` with the commit, host,
device and toolchain they came from. At 4096x4096, float32:

| Kernel | naive original | competent HIP | torch | candidate | vs original | **vs competent** | vs torch |
|---|---|---|---|---|---|---|---|
| `row_softmax` | 2431.9 us | 53.3 us | 40.6 us | **31.7 us** | 76.7x | **1.7x** | 1.3x |
| `layer_norm` | 1739.7 us | 45.7 us | 44.7 us | **31.0 us** | 56.1x | **1.5x** | 1.4x |
| `layer_norm_affine` | 1829.9 us | 46.9 us | 45.7 us | **33.2 us** | 55.2x | **1.4x** | 1.4x |
| `rms_norm` | 1428.3 us | 41.2 us | 43.1 us | **31.2 us** | 45.8x | **1.3x** | 1.4x |
| `rms_norm_affine` | 1578.2 us | 41.4 us | 43.6 us | **32.0 us** | 49.4x | **1.3x** | 1.4x |
| `rope` | 1371.6 us | **45.7 us** | 241.3 us | 53.1 us | 25.8x | **0.9x** | 4.5x |

**The 25x to 77x column is the least interesting one and is close to
meaningless.** It is measured against kernels that launch one thread per block
and leave a 304-CU device idle. Any competent kernel beats them. That is why the
table carries two baselines that can actually win.

**The column that matters is `vs competent`, and it used to read 2.9x.** The
first version of `row_softmax_tuned.cu` made three passes over the row and torch
beat it by 2.3x, so "2.9x against a competently written HIP kernel" was really
2.9x against a mediocre one. Rewriting it as an online softmax, two passes
instead of three, took it from 94.2 us to 53.3 us, and the claim fell to 1.7x.
The LayerNorm baselines moved the same way once they used Welford in a single
pass rather than separate mean and variance passes.

So the honest claim is: **at large shapes the tuned Triton kernels beat a
competently written HIP kernel by 1.3x to 1.7x and torch by 1.3x to 1.4x, and
RoPE loses to a good hand-written kernel outright.** That is a smaller number
than this README used to carry and a much harder one to argue with, and finding
it out cost nothing except being willing to improve the opponent.

RoPE deserves its own sentence: hand-written HIP wins at 0.9x, and the 4.5x
against torch says more about torch having no fused RoPE than about the kernel.
There is no reduction in RoPE, so there is nothing for a tuned implementation to
recover.

Below roughly 16M elements every substitution is a **3x to 5x regression**,
marked `latency-bound` in the tables. The crossover is dispatch cost, measured
rather than assumed: on this box an in-place torch op that does no work costs
**4.9 us** to launch from Python, `torch.softmax` on one row costs **5.5 us**,
and the Triton candidate costs **17 us**. About 12 us is Triton's own launch
path and it is fixed, so it dominates until the kernel has real work to do.

One asymmetry worth stating: native baselines are timed inside the generated C++
driver with device events and never pay Python dispatch, while the candidate and
torch are timed through Python and do. That flatters the native side by roughly
5 us. It does not change the conclusions, because the small-shape gaps are 12 us
and more and the large-shape ones run the other way.

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

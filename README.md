# hipbridge

Recognize CUDA kernels, substitute verified AMD implementations, prove it numerically.

**Status: pre-alpha.** Nothing here has been run on AMD hardware. Claims in this
README are limited to what the test suite actually exercises, and the test suite
runs on CPU.

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

## Layout and the future split

`src/hipbridge/verify/` and `src/hipbridge/kernels/` are kept import-clean so
each can be promoted to its own distribution without a refactor. Promote on
evidence (independent users, independent issues), not on a hunch.

## License

Apache-2.0.

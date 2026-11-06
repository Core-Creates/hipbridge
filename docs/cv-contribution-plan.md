# Proposal: computer-vision kernel coverage for hipbridge

**Author:** Srikanth Arunachalam (@srikantharun) · **Status:** draft, pre-discussion

## Motivation

hipbridge today verifies the transformer-inference row-wise path (softmax,
layer/RMS norm, RoPE, fused). Its own README lists "kernels outside the
transformer inference row-wise path" as not-yet-covered. **Computer-vision
kernels are that gap**, and many of them fit the recognizer patterns that
already exist — so the verification machinery (float64 oracle, ULP budgets,
wavefront-63/64/65 stress) can be reused rather than rebuilt.

This proposes a small, staged CV family, starting with the simplest kernel that
exercises a *different* pattern than the current suites.

## Roadmap (simplest first)

| Kernel | Recognizer pattern | Notes |
|---|---|---|
| **RGB → grayscale** | `elementwise.flat` | Per-pixel weighted sum of 3 channels. No reduction. **This PR scaffolds it.** |
| Resize / bilinear | `row_map` (likely) | Per-output-pixel gather; no accumulation across outputs. |
| 2D convolution (small fixed kernel) | `reduce.tree` / tiled | Shared-memory stencil; reuses the tree-reduction lowering notes. |
| NMS (non-max suppression) | new rule | Control-flow heavy; likely needs a new recognizer + careful oracle. |

## What this branch adds (scaffold only — not yet wired into the suite registry)

- `examples/rgb_to_gray.cu` — CUDA reference. Planar R/G/B, one output per
  thread, no loops/shared/barriers → classifies as `elementwise.flat`.
- `examples/rgb_to_gray_tuned.cu` — the competently hand-written HIP baseline
  (256-thread block), per the `<kernel>_tuned.cu` convention.
- `src/hipbridge/kernels/cv.py` — `rgb_to_gray` Triton implementation, float32
  combine, 64-multiple block, marked `UNVERIFIED` until the harness proves it.

## What's left (the real work, needs maintainer input + AMD hardware)

1. **A `Suite` in `verify/suites.py`.** Proposed literal below. The open design
   question is operands: every current suite transforms **one** tensor, but
   grayscale takes **three** input planes and writes one. `Suite.extras`
   (`Operand`) exists for LayerNorm's gamma/beta — are extra *input* planes the
   intended home, or does the CV family want a small shape/inputs extension?
   **This is the question for the thread before a PR.**

2. **Shapes + inputs.** `shapes.row_wise()` is transformer-shaped. CV wants
   image shapes (e.g. `H×W`, a few channel counts). A `shapes.image()` helper
   would keep the sweep honest.

3. **Oracle.** `lambda r, g, b: (0.299*r + 0.587*g + 0.114*b).double()` —
   trivially a float64 truth, which is ideal for the ULP-budget comparison.

4. **Verify on MI300X (`gfx942`).** Run `hipbridge.verify` for fp32/fp16/bf16
   across the image sweep; commit evidence to `results/`. (Planned on an AMD
   Developer Cloud droplet.)

### Proposed Suite (draft — do not merge until operands are agreed)

```python
RGB_TO_GRAY = Suite(
    name="rgb_to_gray",
    source_file="rgb_to_gray.cu",
    kernel="rgb_to_gray",
    launch=LaunchSpec(
        kernel="rgb_to_gray",
        grid=lambda s: (triton.cdiv(prod(s[0]), 256), 1, 1),
        block=(256, 1, 1),
        scalar_args=lambda s: [prod(s[0])],
        out_shape=lambda s: s[0],
    ),
    oracle=lambda r, g, b: (0.299 * r.double() + 0.587 * g.double() + 0.114 * b.double()),
    portable=lambda r, g, b: 0.299 * r + 0.587 * g + 0.114 * b,
    triton_impl="hipbridge.kernels.cv:rgb_to_gray",
    triton_label="hipbridge.kernels.cv (Triton, AMD-tuned)",
    patterns=(Pattern.ELEMENTWISE,),
    evidence=...,          # a looks_like_elementwise with >=3 input pointers
    usage_call="out = rgb_to_gray(r, g, b)",
    shapes=_image_shapes,  # to be added in verify/shapes.py
    baselines=(tuned_baseline("rgb_to_gray"),),
)
```

## Non-goals (for now)

- Interleaved (HWC) RGB — planar first keeps it strictly elementwise.
- RDNA (32-wide) tuning — follow the project's CDNA-first stance.

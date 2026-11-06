"""Computer-vision kernels, tuned for AMD CDNA.

The first of a planned computer-vision family (grayscale, resize, 2D
convolution, NMS). Grayscale is the simplest: a pure elementwise per-pixel
weighted sum of three channels, so it exercises the elementwise.flat path
rather than any reduction.

Block sizes are multiples of 64 because a CDNA wavefront is 64 wide; a 32-wide
choice wastes half of every wavefront.

UNVERIFIED on AMD hardware. Run hipbridge.verify against this before trusting it.
"""

from __future__ import annotations

import triton
import triton.language as tl

# Rec.601 luma weights. Full precision on purpose: these are constants, so there
# is no reason to round them to the input dtype before the multiply.
#
# Declared tl.constexpr, not plain floats: Triton 3.x refuses to read a plain
# module global from inside an @triton.jit kernel ("can only access global
# variables that are instantiated as constexpr"). As bare floats these compiled
# nowhere, which is why the kernel shipped UNVERIFIED; the verify harness is what
# surfaced it.
_R_WEIGHT = tl.constexpr(0.299)
_G_WEIGHT = tl.constexpr(0.587)
_B_WEIGHT = tl.constexpr(0.114)


@triton.jit
def _rgb_to_gray_kernel(
    r_ptr,
    g_ptr,
    b_ptr,
    out_ptr,
    n,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n

    # Load in whatever precision the caller used, combine in float32. A half
    # precision weighted sum would drop mantissa the reference keeps, and the
    # whole point of the substitute is to be no less accurate than the kernel
    # it replaces.
    r = tl.load(r_ptr + offs, mask=mask, other=0).to(tl.float32)
    g = tl.load(g_ptr + offs, mask=mask, other=0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0).to(tl.float32)

    y = _R_WEIGHT * r + _G_WEIGHT * g + _B_WEIGHT * b

    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def rgb_to_gray(r, g, b):
    """Planar RGB -> single-channel luma (Rec.601), elementwise.

    Takes three equally shaped tensors (the R, G and B planes) and returns one
    tensor of the same shape holding the luma. Shapes are flattened for the
    launch, so any contiguous layout works; the caller keeps whatever 2D/3D
    shape it had on the way out.
    """
    import torch

    if not (r.shape == g.shape == b.shape):
        raise ValueError(
            f"R/G/B planes must match: got {tuple(r.shape)}, "
            f"{tuple(g.shape)}, {tuple(b.shape)}"
        )
    if not (r.is_contiguous() and g.is_contiguous() and b.is_contiguous()):
        r, g, b = r.contiguous(), g.contiguous(), b.contiguous()

    out = torch.empty_like(r)
    n = r.numel()
    if n == 0:
        return out

    # 1024 is 16 wavefronts of 64; plenty of elementwise work per program with
    # no reduction to constrain the tile.
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _rgb_to_gray_kernel[grid](
        r.reshape(-1),
        g.reshape(-1),
        b.reshape(-1),
        out.reshape(-1),
        n,
        BLOCK=BLOCK,
        num_warps=4,
    )
    return out

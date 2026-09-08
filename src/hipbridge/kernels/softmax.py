"""Row-wise softmax, tuned for AMD CDNA.

Block sizes are multiples of 64 because a CDNA wavefront is 64 wide; a 32-wide
choice wastes half of every wavefront.

UNVERIFIED on AMD hardware. Run hipbridge.verify against this before trusting it.
"""

from __future__ import annotations

import triton
import triton.language as tl

from hipbridge.kernels import wide


@triton.jit
def _softmax_rowwise_kernel(
    in_ptr,
    out_ptr,
    in_row_stride,
    in_col_stride,
    out_row_stride,
    out_col_stride,
    n_cols,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    # Load in whatever precision the caller used, reduce in float32. A half
    # precision sum of 4096 exponentials loses most of its mantissa, and the
    # kernel this replaces would not have made that mistake in fp32.
    x = tl.load(
        in_ptr + row * in_row_stride + cols * in_col_stride,
        mask=mask,
        other=-float("inf"),
    )
    x = x.to(tl.float32)

    # Max subtraction for numerical stability. Dropping this is the single most
    # common softmax translation bug and it is silent on small inputs.
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    y = num / den

    tl.store(
        out_ptr + row * out_row_stride + cols * out_col_stride,
        y.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def softmax_rowwise(x):
    """Softmax over the last dimension of a 2D tensor.

    Element strides are passed rather than assumed. Indexing as
    `row * row_stride + col` silently requires `stride(-1) == 1`, which
    nothing asserted and nothing tested, because inputs.py only ever built
    fresh contiguous tensors: `softmax_rowwise(x.t())` returned wrong
    numbers with no error. Triton specialises on arguments equal to 1, so
    the contiguous path is unchanged.
    """
    if x.ndim != 2:
        raise ValueError(f"expected a 2D tensor, got shape {tuple(x.shape)}")

    import torch

    n_rows, n_cols = x.shape
    # Contiguous regardless of what came in. empty_like inherits the input's
    # layout, so a transposed input produced a transposed output and the
    # store was wrong in the same way the load was.
    out = torch.empty((n_rows, n_cols), dtype=x.dtype, device=x.device)

    if x.numel() == 0:
        # next_power_of_2(0) is 1, the mask is all false, and a reduction over
        # an all -inf vector propagates NaN. There is nothing to compute.
        return out
    if n_cols > wide.TILED_ABOVE:
        # A vocabulary softmax is 32k to 128k columns. One block per row
        # cannot hold that, so the row is walked in tiles instead.
        wide.softmax(x, out)
        return out

    block = triton.next_power_of_2(n_cols)
    # 64-wide wavefronts: scale warps with the row so small rows do not
    # under-occupy and large rows do not spill.
    num_warps = 4 if block < 2048 else (8 if block < 8192 else 16)

    _softmax_rowwise_kernel[(n_rows,)](
        x,
        out,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        n_cols,
        BLOCK=block,
        num_warps=num_warps,
    )
    return out

"""Row-wise softmax, tuned for AMD CDNA.

Block sizes are multiples of 64 because a CDNA wavefront is 64 wide; a 32-wide
choice wastes half of every wavefront.

UNVERIFIED on AMD hardware. Run drover.verify against this before trusting it.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _softmax_rowwise_kernel(
    in_ptr,
    out_ptr,
    in_row_stride,
    out_row_stride,
    n_cols,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(in_ptr + row * in_row_stride + cols, mask=mask, other=-float("inf"))

    # Max subtraction for numerical stability. Dropping this is the single most
    # common softmax translation bug and it is silent on small inputs.
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    y = num / den

    tl.store(out_ptr + row * out_row_stride + cols, y, mask=mask)


def softmax_rowwise(x):
    """Softmax over the last dimension of a 2D tensor."""
    if x.ndim != 2:
        raise ValueError(f"expected a 2D tensor, got shape {tuple(x.shape)}")

    import torch

    n_rows, n_cols = x.shape
    out = torch.empty_like(x)

    block = triton.next_power_of_2(n_cols)
    # 64-wide wavefronts: scale warps with the row so small rows do not
    # under-occupy and large rows do not spill.
    num_warps = 4 if block < 2048 else (8 if block < 8192 else 16)

    _softmax_rowwise_kernel[(n_rows,)](
        x,
        out,
        x.stride(0),
        out.stride(0),
        n_cols,
        BLOCK=block,
        num_warps=num_warps,
    )
    return out

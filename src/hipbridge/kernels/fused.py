"""RMSNorm followed by RoPE, in one launch.

The measurement that motivated this: below roughly 16M elements every
substitution in this project is a 3x to 5x regression, and the cause is not the
kernels. Triton's launch path costs about 16 to 20 us and does not shrink with
the work, so a small kernel spends its time being dispatched. Two small kernels
spend it twice.

Fusion is the one answer to that which a library cannot give you. torch has to
keep `rms_norm` and a rotation separable, because it cannot know you will always
call them together; a substitution tool can know, because it is handed the pair.

The maths is unchanged and deliberately so. Normalise by the root mean square,
scale by gamma, then rotate adjacent channel pairs by the position's angle. What
changes is that the row is read once instead of twice and the result is written
once instead of twice, and that the dispatch is paid once.

Channels are read in pairs from the start, so the reduction and the rotation see
the same loaded values: sum(x0^2) + sum(x1^2) is the same sum of squares as a
row-wise pass, taken in an order that happens to leave the pairs in registers.

UNVERIFIED on AMD hardware until hipbridge.verify says otherwise.
"""

from __future__ import annotations

import triton
import triton.language as tl

from hipbridge.kernels import wide


@triton.jit
def _rms_norm_rope_kernel(
    in_ptr,
    gamma_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    in_row_stride,
    in_col_stride,
    out_row_stride,
    out_col_stride,
    gamma_stride,
    cos_row_stride,
    cos_col_stride,
    sin_row_stride,
    sin_col_stride,
    n_cols,
    half,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    mask = i < half

    # Pairs up front: the same loads feed the reduction and the rotation.
    base = in_ptr + row * in_row_stride
    x0 = tl.load(base + (2 * i) * in_col_stride, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(base + (2 * i + 1) * in_col_stride, mask=mask, other=0.0).to(tl.float32)

    ms = (tl.sum(x0 * x0, axis=0) + tl.sum(x1 * x1, axis=0)) / n_cols
    inv = tl.rsqrt(ms + eps)

    g0 = tl.load(gamma_ptr + (2 * i) * gamma_stride, mask=mask, other=0.0).to(tl.float32)
    g1 = tl.load(gamma_ptr + (2 * i + 1) * gamma_stride, mask=mask, other=0.0).to(tl.float32)
    y0 = x0 * inv * g0
    y1 = x1 * inv * g1

    # Each table carries its own strides. Passing cos_tab.stride(0) for both
    # read correct cosines and garbage sines whenever the two were built by
    # different paths, which produces a plausible non-rotation.
    c = tl.load(cos_ptr + row * cos_row_stride + i * cos_col_stride, mask=mask, other=1.0).to(
        tl.float32
    )
    s = tl.load(sin_ptr + row * sin_row_stride + i * sin_col_stride, mask=mask, other=0.0).to(
        tl.float32
    )

    out_ty = out_ptr.dtype.element_ty
    dst = out_ptr + row * out_row_stride
    tl.store(dst + (2 * i) * out_col_stride, (y0 * c - y1 * s).to(out_ty), mask=mask)
    tl.store(dst + (2 * i + 1) * out_col_stride, (y0 * s + y1 * c).to(out_ty), mask=mask)


def rms_norm_rope_rowwise(x, gamma, cos_tab, sin_tab, eps: float = 1e-5):
    """RMSNorm with a learned scale, then rotary embedding, in a single launch."""
    if x.ndim != 2:
        raise ValueError(f"expected a 2D tensor, got shape {tuple(x.shape)}")
    n_rows, n_cols = x.shape
    if n_cols % 2:
        raise ValueError(f"head dimension must be even to pair channels, got {n_cols}")

    half = n_cols // 2
    if gamma.shape != (n_cols,):
        raise ValueError(f"gamma must be ({n_cols},), got {tuple(gamma.shape)}")
    if cos_tab.shape != (n_rows, half) or sin_tab.shape != (n_rows, half):
        raise ValueError(
            f"cos and sin must be ({n_rows}, {half}), got "
            f"{tuple(cos_tab.shape)} and {tuple(sin_tab.shape)}"
        )

    import torch

    # Contiguous regardless of what came in. empty_like inherits the input's
    # layout, so a transposed input produced a transposed output and the
    # store was wrong in the same way the load was.
    out = torch.empty((n_rows, n_cols), dtype=x.dtype, device=x.device)
    if n_cols > wide.TILED_ABOVE:
        # Refused rather than tiled. A rotation operates on a head dimension,
        # which is tens to hundreds of channels; a row this wide is a shape
        # mistake far more often than it is a rotation, and silently spilling
        # is a worse answer than saying so.
        raise ValueError(
            f"head dimension {n_cols} exceeds {wide.TILED_ABOVE}; a rotation "
            "operates on a head dimension, so this is more likely a shape mistake"
        )

    block = max(16, triton.next_power_of_2(half))

    _rms_norm_rope_kernel[(n_rows,)](
        x,
        gamma,
        cos_tab,
        sin_tab,
        out,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        gamma.stride(0),
        cos_tab.stride(0),
        cos_tab.stride(1),
        sin_tab.stride(0),
        sin_tab.stride(1),
        n_cols,
        half,
        eps,
        BLOCK=block,
        num_warps=4 if block < 2048 else 8,
    )
    return out


__all__ = ["rms_norm_rope_rowwise"]

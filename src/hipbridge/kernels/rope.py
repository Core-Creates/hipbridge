"""Rotary position embedding, tuned for AMD CDNA.

Not a reduction. Each output pair depends on the matching input pair and the
angle for that position, so there is nothing to reduce and nothing to lose to
serial accumulation. The accuracy argument that carries softmax and the norms
does not apply here, and saying so is more useful than implying it does: the
case for substituting RoPE is throughput and wavefront-shaped access, not
numerics.

The tables hold cols/2 angles per row, and cols must be even. An odd head
dimension has no pairing and is rejected rather than silently truncated.

UNVERIFIED on AMD hardware until hipbridge.verify says otherwise.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _rope_kernel(
    in_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    in_row_stride,
    in_col_stride,
    out_row_stride,
    out_col_stride,
    cos_row_stride,
    cos_col_stride,
    sin_row_stride,
    sin_col_stride,
    half,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    mask = i < half

    # Interleaved pairs: channel 2i and 2i+1 rotate together.
    # Rotate in float32 even when the data is half: the products cancel, and a
    # cancellation evaluated in fp16 loses bits that were never recoverable.
    base = in_ptr + row * in_row_stride
    x0 = tl.load(base + (2 * i) * in_col_stride, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(base + (2 * i + 1) * in_col_stride, mask=mask, other=0.0).to(tl.float32)
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
    tl.store(dst + (2 * i) * out_col_stride, (x0 * c - x1 * s).to(out_ty), mask=mask)
    tl.store(dst + (2 * i + 1) * out_col_stride, (x0 * s + x1 * c).to(out_ty), mask=mask)


def rope_rowwise(x, cos_tab, sin_tab):
    """Rotate interleaved channel pairs by per-position angles.

    One row is one position, `x` is (positions, head_dim), and the tables are
    (positions, head_dim / 2).
    """
    if x.ndim != 2:
        raise ValueError(f"expected a 2D tensor, got shape {tuple(x.shape)}")
    n_rows, n_cols = x.shape
    if n_cols % 2:
        raise ValueError(f"head dimension must be even to pair channels, got {n_cols}")

    half = n_cols // 2
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
    block = max(16, triton.next_power_of_2(half))

    _rope_kernel[(n_rows,)](
        x,
        cos_tab,
        sin_tab,
        out,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        cos_tab.stride(0),
        cos_tab.stride(1),
        sin_tab.stride(0),
        sin_tab.stride(1),
        half,
        BLOCK=block,
        num_warps=4 if block < 2048 else 8,
    )
    return out


__all__ = ["rope_rowwise"]

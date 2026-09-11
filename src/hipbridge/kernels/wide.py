"""Rows too wide for one block, handled by tiling instead of failing.

Every kernel in this package sizes its block as `next_power_of_2(n_cols)` and
loads a whole row at once. That is the right shape for a hidden dimension and
the wrong shape for a vocabulary: `num_warps` saturates at 16, so at 65536
columns each lane holds 64 elements plus the exponential and sum temporaries,
and Triton either spills catastrophically or fails to compile with an opaque
resource error. A vocabulary softmax is 32k to 128k columns and is the single
most common wide-row kernel in inference, so "the widest thing we support is
4096" was a gap in the shape of the actual use case.

These take the persistent-row form: one block per row still, but the row is
walked in tiles and the reduction is carried in a vector accumulator that is
reduced once at the end. That idiom is deliberate. Loop-carried scalars in
Triton are where type errors hide, and an accumulator of shape [TILE] behaves
exactly like the single-tile kernels these mirror.

Correctness is preserved rather than traded: softmax still subtracts its
maximum, LayerNorm still centres before squaring instead of using the
E[x^2] - E[x]^2 identity that loses precision when the mean dominates, and
accumulation is float32 regardless of the storage type. The cost is re-reading
the row two or three times, which is bandwidth this path was never going to
avoid anyway.

UNVERIFIED until hipbridge.verify says otherwise. Triton publishes no Windows
build, so nothing here executes before it reaches the MI300X.
"""

from __future__ import annotations

import triton
import triton.language as tl

# Defined in the package root, which imports no triton, so that a sweep can be
# built without one. Re-exported here because rope.py and fused.py read it as
# wide.TILED_ABOVE, next to the tiling this module implements.
from hipbridge.kernels import TILED_ABOVE

# Elements per pass. A power of two so the mask arithmetic stays cheap, and
# small enough that the accumulator plus a tile of data fits comfortably.
TILE = 2048


def _warps_for_tile() -> int:
    """A tile is a fixed size, so its occupancy does not depend on the row."""
    return 8


@triton.jit
def _softmax_tiled_kernel(
    in_ptr,
    out_ptr,
    in_row_stride,
    in_col_stride,
    out_row_stride,
    out_col_stride,
    n_cols,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    src = in_ptr + row * in_row_stride
    dst = out_ptr + row * out_row_stride

    # Pass 1: the row maximum, elementwise across tiles then reduced once.
    running_max = tl.full((TILE,), -float("inf"), tl.float32)
    for start in range(0, n_cols, TILE):
        cols = start + tl.arange(0, TILE)
        x = tl.load(src + cols * in_col_stride, mask=cols < n_cols, other=-float("inf"))
        running_max = tl.maximum(running_max, x.to(tl.float32))
    row_max = tl.max(running_max, axis=0)

    # Pass 2: the denominator, with the maximum already subtracted.
    running_sum = tl.zeros((TILE,), tl.float32)
    for start in range(0, n_cols, TILE):
        cols = start + tl.arange(0, TILE)
        mask = cols < n_cols
        x = tl.load(src + cols * in_col_stride, mask=mask, other=-float("inf"))
        running_sum += tl.where(mask, tl.exp(x.to(tl.float32) - row_max), 0.0)
    denom = tl.sum(running_sum, axis=0)

    # Pass 3: read once more and write.
    for start in range(0, n_cols, TILE):
        cols = start + tl.arange(0, TILE)
        mask = cols < n_cols
        x = tl.load(src + cols * in_col_stride, mask=mask, other=-float("inf"))
        y = tl.exp(x.to(tl.float32) - row_max) / denom
        tl.store(dst + cols * out_col_stride, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _rms_norm_tiled_kernel(
    in_ptr,
    gamma_ptr,
    out_ptr,
    in_row_stride,
    in_col_stride,
    out_row_stride,
    out_col_stride,
    gamma_stride,
    n_cols,
    eps,
    HAS_GAMMA: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    src = in_ptr + row * in_row_stride
    dst = out_ptr + row * out_row_stride

    acc = tl.zeros((TILE,), tl.float32)
    for start in range(0, n_cols, TILE):
        cols = start + tl.arange(0, TILE)
        x = tl.load(src + cols * in_col_stride, mask=cols < n_cols, other=0.0).to(tl.float32)
        acc += x * x
    inv = tl.rsqrt(tl.sum(acc, axis=0) / n_cols + eps)

    for start in range(0, n_cols, TILE):
        cols = start + tl.arange(0, TILE)
        mask = cols < n_cols
        x = tl.load(src + cols * in_col_stride, mask=mask, other=0.0).to(tl.float32)
        y = x * inv
        if HAS_GAMMA:
            g = tl.load(gamma_ptr + cols * gamma_stride, mask=mask, other=0.0).to(tl.float32)
            y = y * g
        tl.store(dst + cols * out_col_stride, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _layer_norm_tiled_kernel(
    in_ptr,
    gamma_ptr,
    beta_ptr,
    out_ptr,
    in_row_stride,
    in_col_stride,
    out_row_stride,
    out_col_stride,
    gamma_stride,
    beta_stride,
    n_cols,
    eps,
    HAS_AFFINE: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    src = in_ptr + row * in_row_stride
    dst = out_ptr + row * out_row_stride

    # Pass 1: the mean.
    total = tl.zeros((TILE,), tl.float32)
    for start in range(0, n_cols, TILE):
        cols = start + tl.arange(0, TILE)
        x = tl.load(src + cols * in_col_stride, mask=cols < n_cols, other=0.0).to(tl.float32)
        total += x
    mean = tl.sum(total, axis=0) / n_cols

    # Pass 2: the variance, centred rather than via E[x^2] - E[x]^2. The
    # identity is one pass cheaper and loses most of its precision when the mean
    # is large next to the spread, which is exactly a wide row of activations.
    squares = tl.zeros((TILE,), tl.float32)
    for start in range(0, n_cols, TILE):
        cols = start + tl.arange(0, TILE)
        mask = cols < n_cols
        x = tl.load(src + cols * in_col_stride, mask=mask, other=0.0).to(tl.float32)
        centred = tl.where(mask, x - mean, 0.0)
        squares += centred * centred
    inv = tl.rsqrt(tl.sum(squares, axis=0) / n_cols + eps)

    # Pass 3: write.
    for start in range(0, n_cols, TILE):
        cols = start + tl.arange(0, TILE)
        mask = cols < n_cols
        x = tl.load(src + cols * in_col_stride, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * inv
        if HAS_AFFINE:
            g = tl.load(gamma_ptr + cols * gamma_stride, mask=mask, other=0.0).to(tl.float32)
            b = tl.load(beta_ptr + cols * beta_stride, mask=mask, other=0.0).to(tl.float32)
            y = y * g + b
        tl.store(dst + cols * out_col_stride, y.to(out_ptr.dtype.element_ty), mask=mask)


def softmax(x, out) -> None:
    """Launch the tiled softmax over an already-allocated output."""
    n_rows, n_cols = x.shape
    _softmax_tiled_kernel[(n_rows,)](
        x,
        out,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        n_cols,
        TILE=TILE,
        num_warps=_warps_for_tile(),
    )


def rms_norm(x, out, eps: float, gamma=None) -> None:
    n_rows, n_cols = x.shape
    _rms_norm_tiled_kernel[(n_rows,)](
        x,
        gamma if gamma is not None else x,
        out,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        gamma.stride(0) if gamma is not None else 1,
        n_cols,
        eps,
        HAS_GAMMA=gamma is not None,
        TILE=TILE,
        num_warps=_warps_for_tile(),
    )


def layer_norm(x, out, eps: float, gamma=None, beta=None) -> None:
    n_rows, n_cols = x.shape
    affine = gamma is not None and beta is not None
    _layer_norm_tiled_kernel[(n_rows,)](
        x,
        gamma if affine else x,
        beta if affine else x,
        out,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        gamma.stride(0) if affine else 1,
        beta.stride(0) if affine else 1,
        n_cols,
        eps,
        HAS_AFFINE=affine,
        TILE=TILE,
        num_warps=_warps_for_tile(),
    )


__all__ = ["TILE", "TILED_ABOVE", "layer_norm", "rms_norm", "softmax"]

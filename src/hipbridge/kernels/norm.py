"""Row-wise LayerNorm and RMSNorm, tuned for AMD CDNA.

Block sizes are multiples of 64 because a CDNA wavefront is 64 wide; a 32-wide
choice wastes half of every wavefront.

These are the kernels where the accuracy argument is sharpest. Normalization
sums the row twice over (LayerNorm) or sums its squares (RMSNorm), and a serial
accumulation grows rounding error as O(n) where a pairwise reduction grows it as
O(log n). Squares make it worse: they widen the dynamic range being summed, so
the naive version loses more of it. A translation judged by "does it match the
original" would reject the better kernel here even more firmly than for softmax.

Neither applies affine parameters. Real LayerNorm scales by gamma and shifts by
beta, and hipbridge's harness generates a single input tensor per case, so the
weights would not be covered by the proof. An unproven substitution is worse
than none, so the affine step is left out until the harness can verify it.

UNVERIFIED on AMD hardware until hipbridge.verify says otherwise.
"""

from __future__ import annotations

import triton
import triton.language as tl


def _warps_for(block: int) -> int:
    """64-wide wavefronts: scale with the row so small rows do not under-occupy."""
    return 4 if block < 2048 else (8 if block < 8192 else 16)


@triton.jit
def _layer_norm_kernel(
    in_ptr,
    out_ptr,
    in_row_stride,
    out_row_stride,
    n_cols,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    # Zero-fill rather than -inf: these values enter a sum, not a maximum.
    x = tl.load(in_ptr + row * in_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    n = tl.sum(mask.to(tl.float32), axis=0)

    mean = tl.sum(x, axis=0) / n
    centred = tl.where(mask, x - mean, 0.0)
    var = tl.sum(centred * centred, axis=0) / n

    y = centred * tl.rsqrt(var + eps)
    tl.store(out_ptr + row * out_row_stride + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _rms_norm_kernel(
    in_ptr,
    out_ptr,
    in_row_stride,
    out_row_stride,
    n_cols,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(in_ptr + row * in_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    n = tl.sum(mask.to(tl.float32), axis=0)

    ms = tl.sum(x * x, axis=0) / n
    y = x * tl.rsqrt(ms + eps)
    tl.store(out_ptr + row * out_row_stride + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _layer_norm_affine_kernel(
    in_ptr,
    gamma_ptr,
    beta_ptr,
    out_ptr,
    in_row_stride,
    out_row_stride,
    n_cols,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(in_ptr + row * in_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    n = tl.sum(mask.to(tl.float32), axis=0)

    mean = tl.sum(x, axis=0) / n
    centred = tl.where(mask, x - mean, 0.0)
    var = tl.sum(centred * centred, axis=0) / n

    g = tl.load(gamma_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(beta_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = centred * tl.rsqrt(var + eps) * g + b

    tl.store(out_ptr + row * out_row_stride + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


def layer_norm_affine_rowwise(x, gamma, beta, eps: float = 1e-5):
    """LayerNorm with learned scale and shift, the form a transformer uses.

    The weights are per-column, so a row of `n_cols` takes gamma and beta of
    length `n_cols`. This is the kernel that made the harness carry more than
    one input: proving the normalisation alone and then substituting something
    that also multiplies by weights would prove one program and ship another.
    """
    if x.ndim != 2:
        raise ValueError(f"expected a 2D tensor, got shape {tuple(x.shape)}")
    if gamma.shape != (x.shape[-1],) or beta.shape != (x.shape[-1],):
        raise ValueError(
            f"gamma and beta must be ({x.shape[-1]},), got "
            f"{tuple(gamma.shape)} and {tuple(beta.shape)}"
        )

    import torch

    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    block = triton.next_power_of_2(n_cols)

    _layer_norm_affine_kernel[(n_rows,)](
        x,
        gamma,
        beta,
        out,
        x.stride(0),
        out.stride(0),
        n_cols,
        eps,
        BLOCK=block,
        num_warps=_warps_for(block),
    )
    return out


def _launch(kernel, x, eps):
    if x.ndim != 2:
        raise ValueError(f"expected a 2D tensor, got shape {tuple(x.shape)}")

    import torch

    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    block = triton.next_power_of_2(n_cols)

    kernel[(n_rows,)](
        x,
        out,
        x.stride(0),
        out.stride(0),
        n_cols,
        eps,
        BLOCK=block,
        num_warps=_warps_for(block),
    )
    return out


def layer_norm_rowwise(x, eps: float = 1e-5):
    """LayerNorm over the last dimension of a 2D tensor, without affine terms."""
    return _launch(_layer_norm_kernel, x, eps)


def rms_norm_rowwise(x, eps: float = 1e-5):
    """RMSNorm over the last dimension of a 2D tensor, without affine terms."""
    return _launch(_rms_norm_kernel, x, eps)


__all__ = [
    "layer_norm_affine_rowwise",
    "layer_norm_rowwise",
    "rms_norm_affine_rowwise",
    "rms_norm_rowwise",
]


@triton.jit
def _rms_norm_affine_kernel(
    in_ptr,
    gamma_ptr,
    out_ptr,
    in_row_stride,
    out_row_stride,
    n_cols,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(in_ptr + row * in_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    n = tl.sum(mask.to(tl.float32), axis=0)

    ms = tl.sum(x * x, axis=0) / n
    g = tl.load(gamma_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * tl.rsqrt(ms + eps) * g

    tl.store(out_ptr + row * out_row_stride + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


def rms_norm_affine_rowwise(x, gamma, eps: float = 1e-5):
    """RMSNorm with a learned scale, the form most current LLMs use."""
    if x.ndim != 2:
        raise ValueError(f"expected a 2D tensor, got shape {tuple(x.shape)}")
    if gamma.shape != (x.shape[-1],):
        raise ValueError(f"gamma must be ({x.shape[-1]},), got {tuple(gamma.shape)}")

    import torch

    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    block = triton.next_power_of_2(n_cols)

    _rms_norm_affine_kernel[(n_rows,)](
        x,
        gamma,
        out,
        x.stride(0),
        out.stride(0),
        n_cols,
        eps,
        BLOCK=block,
        num_warps=_warps_for(block),
    )
    return out

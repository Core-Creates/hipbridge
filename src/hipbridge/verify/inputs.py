"""Input generation for differential testing.

Inputs are chosen to break implementations, not to confirm them. A softmax that
drops its max-subtraction pass is bitwise correct on N(0,1) and catastrophically
wrong on logits of magnitude 100, so the default sweep includes both.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class Distribution(str, Enum):
    """Input families, each targeting a specific class of bug."""

    NORMAL = "normal"  # baseline, catches gross errors only
    LARGE = "large"  # magnitude ~1e2: overflows unstable exp()
    TINY = "tiny"  # near denormal: flushes to zero incorrectly
    MIXED_SIGN = "mixed_sign"  # catches abs()/sign handling
    CONSTANT = "constant"  # every element equal: degenerate reductions
    MONOTONIC = "monotonic"  # ordered: catches index and stride errors
    SPARSE = "sparse"  # mostly zeros: catches masked accumulation
    WITH_INF = "with_inf"  # +/-inf present: catches inf - inf -> nan
    # What a trained scale actually looks like: clustered near 1, with a few
    # zeros and a few negatives. Deliberately NOT in DEFAULT_SWEEP, because it
    # is a poor input to a kernel and a realistic weight, and those are
    # different jobs. See the note on WEIGHT_FOR below.
    WEIGHT = "weight"


# Distributions that are safe for any elementwise or reduction kernel.
DEFAULT_SWEEP: tuple[Distribution, ...] = (
    Distribution.NORMAL,
    Distribution.LARGE,
    Distribution.TINY,
    Distribution.MIXED_SIGN,
    Distribution.CONSTANT,
    Distribution.MONOTONIC,
    Distribution.SPARSE,
)


class Layout(str, Enum):
    """How a tensor is laid out in memory, as distinct from what is in it.

    Every kernel here indexes as `row * row_stride + col * col_stride`, and
    until those strides were passed the second factor was assumed to be 1.
    Nothing could catch that, because generate() only ever produced fresh
    contiguous tensors: the assumption and the test data made the same
    assumption.

    PADDED is the one that matters most. A kernel written for rows padded to a
    hardware boundary has `row_stride > n_cols`, and proving it against
    unpacked data proves a different program from the one that runs.
    """

    CONTIGUOUS = "contiguous"
    TRANSPOSED = "transposed"  # a view of its own transpose: col stride > 1
    SLICED = "sliced"  # every other column: col stride == 2
    PADDED = "padded"  # rows wider than n_cols: row stride > n_cols


# Layouts worth sweeping beyond the contiguous case. Kept short because each one
# multiplies cases, and the harness runs them as a small separate pass rather
# than crossing them with distributions and precisions.
NON_CONTIGUOUS: tuple[Layout, ...] = (Layout.PADDED, Layout.TRANSPOSED, Layout.SLICED)


def relayout(t, layout: Layout):
    """Return a tensor holding the same values in a different memory layout.

    The values are identical by construction, so a kernel that reads its strides
    correctly cannot tell these apart, and one that does not fails on the very
    first case.
    """
    import torch

    if layout is Layout.CONTIGUOUS:
        return t
    if t.ndim < 2 and layout is Layout.TRANSPOSED:
        # A vector has no transpose. SLICED is what gives a 1-D weight a stride
        # other than 1, which is what exercises gamma_stride.
        return t
    if layout is Layout.TRANSPOSED:
        out = torch.empty(tuple(reversed(t.shape)), dtype=t.dtype, device=t.device).t()
        out.copy_(t)
        return out
    if layout is Layout.SLICED:
        wide = torch.empty((*t.shape[:-1], t.shape[-1] * 2), dtype=t.dtype, device=t.device)
        view = wide[..., ::2]
        view.copy_(t)
        return view
    if layout is Layout.PADDED:
        wide = torch.empty((*t.shape[:-1], t.shape[-1] + 8), dtype=t.dtype, device=t.device)
        view = wide[..., : t.shape[-1]]
        view.copy_(t)
        return view
    raise ValueError(f"unknown layout {layout}")


@dataclass(frozen=True)
class InputSpec:
    shape: tuple[int, ...]
    dtype: torch.dtype = torch.float32
    distribution: Distribution = Distribution.NORMAL
    seed: int = 0
    layout: Layout = Layout.CONTIGUOUS

    def describe(self) -> str:
        dims = "x".join(str(d) for d in self.shape)
        described = f"{dims}/{str(self.dtype).removeprefix('torch.')}/{self.distribution.value}"
        if self.layout is not Layout.CONTIGUOUS:
            described += f"/{self.layout.value}"
        return described


def generate(spec: InputSpec, device: str = "cpu") -> torch.Tensor:
    """Build one deterministic input tensor from a spec."""
    g = torch.Generator(device="cpu").manual_seed(spec.seed)
    shape = spec.shape
    d = spec.distribution

    if d is Distribution.NORMAL:
        t = torch.randn(shape, generator=g)
    elif d is Distribution.LARGE:
        t = torch.randn(shape, generator=g) * 100.0
    elif d is Distribution.TINY:
        # Scaled to the working precision rather than fixed at 1e-30, which is
        # far below what half precision can represent: in fp16 every value would
        # flush to zero and the case would silently stop testing anything. This
        # lands just above the smallest normal for the dtype in use.
        t = torch.randn(shape, generator=g) * (torch.finfo(spec.dtype).tiny * 16)
    elif d is Distribution.MIXED_SIGN:
        t = torch.randn(shape, generator=g)
        t[t.abs() < 0.5] = 0.0
        t = t * torch.where(torch.rand(shape, generator=g) > 0.5, 1.0, -1.0)
    elif d is Distribution.CONSTANT:
        t = torch.full(shape, 3.5)
    elif d is Distribution.MONOTONIC:
        t = torch.arange(int(torch.tensor(shape).prod()), dtype=torch.float32).reshape(shape)
        t = t / max(t.numel(), 1)
    elif d is Distribution.SPARSE:
        t = torch.randn(shape, generator=g)
        t[torch.rand(shape, generator=g) < 0.9] = 0.0
    elif d is Distribution.WEIGHT:
        # A learned gamma sits near 1. The zeros matter because a channel that
        # has been switched off multiplies a whole column away, and the
        # negatives matter because a kernel that assumes a positive scale (by
        # folding it into an abs or a rsqrt, say) is wrong on them.
        t = 1.0 + 0.1 * torch.randn(shape, generator=g)
        picks = torch.rand(shape, generator=g)
        t[picks < 0.05] = 0.0
        t[picks > 0.95] *= -1.0
    elif d is Distribution.WITH_INF:
        t = torch.randn(shape, generator=g)
        flat = t.reshape(-1)
        if flat.numel() >= 2:
            flat[0] = float("inf")
            flat[1] = float("-inf")
    else:  # pragma: no cover - Distribution is exhaustive
        raise ValueError(f"unhandled distribution {d}")

    # Layout last, so the values are decided by the distribution and the memory
    # arrangement by the layout, and the two cannot interfere.
    return relayout(t.to(dtype=spec.dtype, device=device), spec.layout)


# How an operand's distribution follows the primary case's.
#
# Weights inherited the input's distribution, so gamma was drawn from the same
# adversarial sweep as the data: WITH_INF gamma against WITH_INF input was
# covered and a realistic gamma near 1.0 was not. That is backwards, since the
# realistic pairing is the one every user hits.
#
# It is also not simply "always use WEIGHT", because hostile weights do break
# kernels and dropping them would trade one blind spot for another. So the
# ordinary case gets realistic weights, and every adversarial input keeps
# adversarial weights alongside it. Both are covered, and which is which is
# decided here rather than left to whatever the primary happened to be.
WEIGHT_FOR: dict[Distribution, Distribution] = {
    Distribution.NORMAL: Distribution.WEIGHT,
}


def weight_distribution(primary: Distribution) -> Distribution:
    """The distribution an operand should use, given the primary case's."""
    return WEIGHT_FOR.get(primary, primary)


__all__ = [
    "DEFAULT_SWEEP",
    "NON_CONTIGUOUS",
    "WEIGHT_FOR",
    "Distribution",
    "InputSpec",
    "Layout",
    "generate",
    "relayout",
    "weight_distribution",
]

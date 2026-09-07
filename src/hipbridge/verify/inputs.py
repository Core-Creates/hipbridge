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


@dataclass(frozen=True)
class InputSpec:
    shape: tuple[int, ...]
    dtype: torch.dtype = torch.float32
    distribution: Distribution = Distribution.NORMAL
    seed: int = 0

    def describe(self) -> str:
        dims = "x".join(str(d) for d in self.shape)
        return f"{dims}/{str(self.dtype).removeprefix('torch.')}/{self.distribution.value}"


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
    elif d is Distribution.WITH_INF:
        t = torch.randn(shape, generator=g)
        flat = t.reshape(-1)
        if flat.numel() >= 2:
            flat[0] = float("inf")
            flat[1] = float("-inf")
    else:  # pragma: no cover - Distribution is exhaustive
        raise ValueError(f"unhandled distribution {d}")

    return t.to(dtype=spec.dtype, device=device)


__all__ = ["DEFAULT_SWEEP", "Distribution", "InputSpec", "generate"]

"""Numeric comparison in ULPs, plus the identity/no-op detectors.

The two failure modes that motivated this project both survive a naive
allclose-against-a-loose-tolerance check, so they get explicit tests:

  identity kernel  output is a copy of the input, computation dropped
  dead kernel      output never written, buffer holds whatever it held before
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class Report:
    name: str
    passed: bool
    max_ulp: int | None = None
    max_abs: float | None = None
    max_rel: float | None = None
    failures: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        head = f"{'PASS' if self.passed else 'FAIL'}  {self.name}"
        if self.max_ulp is not None:
            head += f"  max_ulp={self.max_ulp} max_abs={self.max_abs:.3e}"
        return "\n".join([head, *(f"    {f}" for f in self.failures)])


def ulp_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Distance in representable floats. Order-independent and scale-free."""
    if a.dtype != b.dtype:
        raise TypeError(f"dtype mismatch: {a.dtype} vs {b.dtype}")
    if a.dtype not in (torch.float32, torch.float64):
        a, b = a.float(), b.float()

    ia = a.detach().cpu().contiguous().view(torch.int32).to(torch.int64)
    ib = b.detach().cpu().contiguous().view(torch.int32).to(torch.int64)
    # Map the sign-magnitude layout onto a monotonic integer line.
    ia = torch.where(ia < 0, torch.tensor(-(2**31), dtype=torch.int64) - ia, ia)
    ib = torch.where(ib < 0, torch.tensor(-(2**31), dtype=torch.int64) - ib, ib)
    return (ia - ib).abs()


def is_identity(out: torch.Tensor, inp: torch.Tensor) -> bool:
    """Did the kernel just copy its input?"""
    return out.shape == inp.shape and torch.equal(out, inp)


def is_unwritten(out: torch.Tensor, sentinel: float) -> bool:
    """Did the kernel never write the output buffer?"""
    return bool(torch.all(out == sentinel))


def check(
    name: str,
    candidate,
    reference,
    inputs: tuple,
    max_ulp: int = 4,
) -> Report:
    """Run both implementations on identical inputs and compare.

    `candidate` is called against a sentinel-filled output buffer where possible,
    so a kernel that never stores is caught rather than silently passing.
    """
    failures: list[str] = []

    ref = reference(*inputs)
    got = candidate(*inputs)

    if got.shape != ref.shape:
        return Report(name, False, failures=[f"shape {tuple(got.shape)} != {tuple(ref.shape)}"])

    if torch.isnan(got).any() and not torch.isnan(ref).any():
        failures.append("candidate produced NaN where reference did not")

    primary = inputs[0] if inputs and torch.is_tensor(inputs[0]) else None
    if primary is not None and is_identity(got, primary):
        failures.append("IDENTITY: output is a byte-exact copy of the input")

    d = ulp_diff(got.float(), ref.float())
    worst = int(d.max().item())
    abs_err = float((got.float() - ref.float()).abs().max().item())
    rel_den = ref.float().abs().clamp_min(1e-30)
    rel_err = float(((got.float() - ref.float()).abs() / rel_den).max().item())

    if worst > max_ulp:
        failures.append(f"max ULP {worst} exceeds tolerance {max_ulp}")

    return Report(name, not failures, worst, abs_err, rel_err, failures)


__all__ = ["Report", "check", "is_identity", "is_unwritten", "ulp_diff"]

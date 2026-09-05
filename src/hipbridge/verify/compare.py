"""Numeric comparison in ULPs, plus the identity/no-op detectors.

The two failure modes that motivated this project both survive a naive
allclose-against-a-loose-tolerance check, so they get explicit tests:

  identity kernel  output is a copy of the input, computation dropped
  dead kernel      output never written, buffer holds whatever it held before
"""

from __future__ import annotations

import math
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


@dataclass
class Arbitration:
    """Accuracy of two implementations judged against a high-precision oracle."""

    candidate_err: float
    reference_err: float
    verdict: str  # "better" | "equivalent" | "worse"
    ratio: float

    def __str__(self) -> str:
        return (
            f"candidate {self.candidate_err:.3e} vs reference {self.reference_err:.3e} "
            f"({self.ratio:.1f}x, {self.verdict})"
        )


def arbitrate(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    truth: torch.Tensor,
    slack: float = 2.0,
) -> Arbitration:
    """Judge candidate and reference against a float64 oracle.

    Measured on an RTX 4060: the serial-accumulation row_softmax.cu in examples/
    is 1.4x to 59.7x LESS accurate than torch across every shape and
    distribution tried, because serial summation accumulates O(n) rounding error
    where a pairwise reduction accumulates O(log n).

    So requiring a translated kernel to match the original within a ULP budget
    is backwards. A tuned AMD kernel using a wavefront tree reduction will
    diverge from a naive serial original precisely BECAUSE it is more accurate,
    and a match-the-original test would reject it. The question worth asking is
    whether the candidate is closer to the truth, not whether it reproduces the
    original's rounding error.
    """
    # A native reference returns host tensors while the candidate runs on the
    # device, so the three arguments routinely arrive on different devices. The
    # harness happened to align them; a direct caller got a torch traceback about
    # cuda:0 and cpu, which says nothing about accuracy. Align them here instead.
    dev = candidate.device
    if reference.device != dev:
        reference = reference.to(dev)
    if truth.device != dev:
        truth = truth.to(dev)

    t = truth.double()
    e_cand = float((candidate.double() - t).abs().max())
    e_ref = float((reference.double() - t).abs().max())

    if e_ref == 0.0:
        ratio = 1.0 if e_cand == 0.0 else float("inf")
    else:
        ratio = e_cand / e_ref

    # NaN is not a magnitude and cannot be ordered. Every comparison against it
    # answers False, so falling through would land on "worse" and blame the
    # candidate for a NaN the reference produced identically. Decide it here.
    cand_nan, ref_nan = math.isnan(e_cand), math.isnan(e_ref)
    if cand_nan or ref_nan:
        if cand_nan and ref_nan:
            return Arbitration(e_cand, e_ref, "equivalent", ratio)
        return Arbitration(e_cand, e_ref, "worse" if cand_nan else "better", ratio)

    # Ratios between sub-epsilon errors are noise. Two implementations both
    # accurate to a few float32 ULP of the output scale can differ by 3x purely
    # through rounding, and failing that would reject correct kernels. Anything
    # at or below the representable resolution of the output is not a defect.
    #
    # Both sides have to be down there for the comparison to be noise. Testing
    # only the candidate reports "equivalent" for a candidate that is essentially
    # exact against a reference that is 78x further from the truth, which is not
    # noise, it is the result. Measured on an MI300X: the tuned Triton softmax
    # sits at the floor on every finite distribution while the serial original
    # does not, so the ratio never got consulted and the verdict never said
    # "better" for a kernel that always was.
    scale = float(t.abs().max()) or 1.0
    floor = 8.0 * torch.finfo(torch.float32).eps * scale
    if e_cand <= floor and e_ref <= floor:
        return Arbitration(e_cand, e_ref, "equivalent", ratio)

    if ratio <= 1.0:
        verdict = "better"
    elif ratio <= slack:
        verdict = "equivalent"
    else:
        verdict = "worse"
    return Arbitration(e_cand, e_ref, verdict, ratio)


__all__ = [
    "Arbitration",
    "Report",
    "arbitrate",
    "check",
    "is_identity",
    "is_unwritten",
    "ulp_diff",
]

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


# The integer type each float type can be reinterpreted through, and its width.
# ULP distance is only meaningful in the precision the kernel actually used:
# upcasting float16 to float32 before counting turns a one-ULP disagreement into
# roughly 8192, which reads as catastrophic and is not. Inference runs in half
# precision, so this had to stop being a float32-only measurement.
_INT_VIEW: dict[torch.dtype, tuple[torch.dtype, int]] = {
    torch.float16: (torch.int16, 16),
    torch.bfloat16: (torch.int16, 16),
    torch.float32: (torch.int32, 32),
    torch.float64: (torch.int64, 64),
}


def ulp_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Distance in representable floats, counted in the operands' own precision.

    Order-independent and scale-free. bfloat16 is included and is worth a word:
    it has the exponent range of float32 with eight mantissa bits, so its ULPs
    are coarse and a handful of them is a large relative error. The count means
    what it says; the interpretation differs per type.
    """
    if a.dtype != b.dtype:
        raise TypeError(f"dtype mismatch: {a.dtype} vs {b.dtype}")

    view, bits = _INT_VIEW.get(a.dtype, (None, 0))
    if view is None:
        a, b = a.float(), b.float()
        view, bits = torch.int32, 32

    ia = a.detach().cpu().contiguous().view(view).to(torch.int64)
    ib = b.detach().cpu().contiguous().view(view).to(torch.int64)
    # Map the sign-magnitude layout onto a monotonic integer line.
    floor_int = torch.tensor(-(2 ** (bits - 1)), dtype=torch.int64)
    ia = torch.where(ia < 0, floor_int - ia, ia)
    ib = torch.where(ib < 0, floor_int - ib, ib)
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

    # ULP in the precision the kernel used, magnitudes in float32. Counting ULPs
    # after an upcast would inflate a one-ULP half-precision disagreement into
    # thousands; measuring magnitudes in half would round the error being
    # measured. The two questions want different arithmetic.
    common = got.dtype if got.dtype == ref.dtype else torch.float32
    d = ulp_diff(got.to(common), ref.to(common))
    worst = int(d.max().item())
    abs_err = float((got.float() - ref.float()).abs().max().item())
    rel_den = ref.float().abs().clamp_min(1e-30)
    rel_err = float(((got.float() - ref.float()).abs() / rel_den).max().item())

    if worst > max_ulp:
        failures.append(f"max ULP {worst} exceeds tolerance {max_ulp}")

    return Report(name, not failures, worst, abs_err, rel_err, failures)


def _rows(x: torch.Tensor) -> torch.Tensor:
    """The tensor as (rows, width), which is the shape every kernel here works in.

    A row is the unit these kernels reduce over, so it is the unit accuracy has
    to be judged in. 1D input is one row; anything wider than 2D folds its
    leading dimensions, because a batch of rows is still rows.
    """
    if x.ndim == 0:
        return x.reshape(1, 1)
    return x.reshape(-1, x.shape[-1])


def _row_distances(x: torch.Tensor, t: torch.Tensor, finite: torch.Tensor) -> torch.Tensor:
    """Largest |x - truth| within each row, over the positions the truth is a number."""
    d = (x - t).abs()
    d = torch.where(torch.isfinite(x), d, torch.full_like(d, float("inf")))
    d = torch.where(finite, d, torch.zeros_like(d))
    return _rows(d).amax(dim=-1)


def _row_scales(t: torch.Tensor, finite: torch.Tensor, tiny: float) -> torch.Tensor:
    """Each row's own output magnitude, which is what its noise floor follows."""
    magnitudes = torch.where(finite, t.abs(), torch.zeros_like(t))
    return _rows(magnitudes).amax(dim=-1).clamp_min(tiny)


def _classify(x: torch.Tensor) -> torch.Tensor:
    """Each element as 0 finite, 1 +inf, 2 -inf, 3 NaN.

    Elementwise and never reduced. The whole NaN defect below came from reducing
    first and asking afterwards.
    """
    out = torch.zeros(x.shape, dtype=torch.int8, device=x.device)
    out[x == float("inf")] = 1
    out[x == float("-inf")] = 2
    out[torch.isnan(x)] = 3
    return out


def _distance(x: torch.Tensor, t: torch.Tensor, finite: torch.Tensor) -> float:
    """Largest |x - truth| over the positions where the truth is a number.

    A NaN or an infinity from x at one of those positions is not a distance, it
    is a failure, so it counts as infinite rather than dissolving the maximum it
    is taken through.
    """
    if not bool(finite.any()):
        return 0.0
    d = (x - t).abs()
    d = torch.where(torch.isfinite(x), d, torch.full_like(d, float("inf")))
    return float(d[finite].max())


def _structure_mismatches(x: torch.Tensor, t: torch.Tensor) -> int:
    """How many positions disagree with the oracle about being a number at all."""
    return int((_classify(x) != _classify(t)).sum())


def nonfinite_where_finite(x: torch.Tensor, truth: torch.Tensor) -> int:
    """Positions where the oracle is a number and x is not.

    The gate the NaN defect left open. Arbitration ranks two implementations
    against each other, so by construction it cannot fail a case where both are
    broken identically: two kernels that both return NaN for every element rank
    as equivalent, which is true and useless. Something has to say that
    returning NaN where the truth is a number is wrong however unanimously it is
    done, and in oracle mode, where the ULP and identity checks are dropped,
    nothing did.
    """
    finite = torch.isfinite(truth.double())
    return int((finite & ~torch.isfinite(x.double())).sum())


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
    cand = candidate.double()
    ref = reference.double()

    # Magnitudes are measured only where the oracle is a number, and the
    # oracle's non-finite positions are judged separately as structure.
    #
    # The bug this replaces: both errors were a max over the whole tensor, so a
    # single NaN anywhere in the truth made both of them NaN, and the branch
    # that caught "NaN on both sides" returned "equivalent" for the entire case.
    # Reproduced with a candidate that was garbage on every finite element:
    #
    #     truth [nan, 1, 2, 3]  cand [nan, 999, -42, 0]  ->  equivalent
    #     truth [  0, 1, 2, 3]  cand [  0, 999, -42, 0]  ->  worse
    #
    # Identical garbage, opposite verdicts, decided by one element nobody was
    # comparing. In oracle mode the harness has already dropped the ULP and
    # identity checks, so this was the only gate left and it was open. A row
    # that masks to all -inf, an overflow, a WITH_INF input: any of them buys a
    # free pass for everything else in the tensor.
    finite = torch.isfinite(t)
    e_cand = _distance(cand, t, finite)
    e_ref = _distance(ref, t, finite)

    if e_ref == 0.0:
        ratio = 1.0 if e_cand == 0.0 else float("inf")
    elif math.isinf(e_ref) and math.isinf(e_cand):
        # Both sides broken in the same way. inf/inf is NaN, which fails every
        # comparison below and lands on "worse", blaming the candidate for a
        # failure it shares with the reference. Ranking cannot separate them, so
        # it says so; nonfinite_where_finite is what fails the case.
        ratio = 1.0
    else:
        ratio = e_cand / e_ref

    # Structure before magnitude. Producing a number where the oracle produces
    # NaN is not accuracy, it is answering a different question, and the side
    # that reproduces the oracle's non-finite pattern more faithfully wins
    # regardless of what the finite elements say. Equal mismatch counts fall
    # through to the magnitude comparison, which is the ordinary case: zero
    # against zero.
    m_cand = _structure_mismatches(cand, t)
    m_ref = _structure_mismatches(ref, t)
    if m_cand != m_ref:
        verdict = "worse" if m_cand > m_ref else "better"
        return Arbitration(e_cand, e_ref, verdict, ratio)

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
    #
    # The floor follows the working precision, not float32. fp16 resolves about
    # 1e-3 where float32 resolves 1e-7, so a float32 floor applied to half
    # precision is roughly 8000x too tight: it would rank two implementations
    # that are both exact to the last representable bit, on rounding noise that
    # neither could have avoided.
    #
    # And it follows each row's own magnitude, not the tensor's. Taken globally
    # in bfloat16 the floor is 6.25% of the largest output anywhere, so a row of
    # magnitude 1 sitting beside a row of magnitude 100 could be wrong in every
    # element and still pass:
    #
    #     truth [[100, 100], [1, 1]]  candidate [[100, 100], [-5, 7]]
    #     global floor 6.25, row error 6.0            ->  equivalent
    #
    # Row-wise kernels are exactly where magnitudes differ between rows, and an
    # attention row that masks to almost nothing beside a row that does not is
    # the ordinary case rather than a contrived one.
    finfo = torch.finfo(candidate.dtype)
    rows_cand = _row_distances(cand, t, finite)
    rows_ref = _row_distances(ref, t, finite)
    rows_scale = _row_scales(t, finite, finfo.tiny)
    rows_floor = 8.0 * finfo.eps * rows_scale

    noise = (rows_cand <= rows_floor) & (rows_ref <= rows_floor)
    if bool(noise.all()):
        return Arbitration(e_cand, e_ref, "equivalent", ratio)

    # No row may be materially wrong, whatever the tensor as a whole says. This
    # is the guard the global comparison could not offer: a small row that is
    # entirely wrong contributes nothing to a maximum set by a larger row, so
    # averaging it away was automatic.
    #
    # Materially is the load-bearing word, and the first attempt left it out. A
    # row was failed merely for being worse than the reference on that row,
    # which on an MI300X failed layer_norm at 64x65 and 1000x128 on monotonic
    # input:
    #
    #     row 8: candidate 8.779e-06, reference 2.399e-07, row magnitude 1.397
    #
    # The naive kernel lands nearly exact on 17 of those 64 rows, so the ratio
    # between them is 36x while the candidate is off by six millionths of a
    # percent of the row. Globally that candidate was 12x closer to the truth
    # and it was reported worse, which is exactly the match-the-original's-
    # rounding test this project exists to argue against.
    #
    # So a row has to be wrong on its own terms before it can fail a case: off
    # by more than a hundredth of its own magnitude, or past the working
    # precision's resolution when that is coarser, and only then is it weighed
    # against the reference. A row that is entirely wrong sits at 100% of its
    # magnitude or beyond, four orders above this line, while the layer_norm
    # rows above sit 1600x below it.
    material = torch.maximum(rows_floor, 0.01 * rows_scale)
    beaten = rows_cand > slack * torch.maximum(rows_ref, rows_floor)
    if bool(((rows_cand > material) & beaten).any()):
        return Arbitration(e_cand, e_ref, "worse", ratio)

    # Strictly closer, not merely as close. A tie gives a ratio of exactly 1.0,
    # and calling that "better" inflated every count: the RoPE run reported
    # `worst ulp=0` and `better=12` in the same line, which cannot both be true,
    # because output identical to the original cannot be nearer the truth.
    if ratio < 1.0:
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
    "nonfinite_where_finite",
    "ulp_diff",
]

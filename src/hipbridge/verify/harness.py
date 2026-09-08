"""Differential verification harness.

Sweeps a candidate implementation against a reference across shapes and input
distributions, and fails loudly on the specific ways a translated kernel goes
wrong:

  identity     output is a copy of the input, computation dropped
  unwritten    output buffer never stored to
  nondeterministic  same input, different answer across runs
  numeric      answers differ beyond the ULP budget

Build this before the translator. It is the only thing that can tell you a
substitution is correct, and it is useful on its own to anyone writing kernels
by hand.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from hipbridge.verify.compare import (
    Arbitration,
    arbitrate,
    check,
    is_identity,
    nonfinite_where_finite,
)
from hipbridge.verify.inputs import DEFAULT_SWEEP, Distribution, InputSpec, generate
from hipbridge.verify.reference import Reference, TorchReference


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bitwise equality, treating NaN in the same position as equal.

    torch.equal reports NaN != NaN, which would flag a kernel that reliably
    produces NaN as nondeterministic. That is a property of IEEE 754 comparison,
    not of the kernel.
    """
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    both_nan = torch.isnan(a) & torch.isnan(b)
    return bool(((a == b) | both_nan).all())


@dataclass
class CaseResult:
    label: str
    passed: bool
    max_ulp: int | None = None
    failures: list[str] = field(default_factory=list)
    arbitration: Arbitration | None = None
    # ULP is scale-free and the right metric for softmax, whose outputs are
    # positive and O(1). It is the wrong one for anything centred on zero:
    # LayerNorm outputs straddle zero, where +1e-9 and -1e-9 are a hair apart in
    # magnitude and 1.7 billion ULP apart on the integer line. A committed report
    # showing only ULP would read as a catastrophe. Carry the absolute error too.
    max_abs: float | None = None

    def as_dict(self) -> dict:
        return {
            "case": self.label,
            "passed": self.passed,
            "max_ulp": self.max_ulp,
            "max_abs": self.max_abs,
            "failures": list(self.failures),
            "arbitration": self.arbitration.as_dict() if self.arbitration else None,
        }

    def __str__(self) -> str:
        mark = "pass" if self.passed else "FAIL"
        ulp = f" ulp={self.max_ulp}" if self.max_ulp is not None else ""
        acc = f" [{self.arbitration.verdict}]" if self.arbitration else ""
        head = f"  {mark}  {self.label}{ulp}{acc}"
        return "\n".join([head, *(f"        {f}" for f in self.failures)])


@dataclass
class Summary:
    name: str
    results: list[CaseResult] = field(default_factory=list)
    skipped_reason: str | None = None
    # Why the reference was refused as a baseline, when it was. Distinct from a
    # skip, which means nothing ran and nothing is claimed, and from a case
    # failure, which is a result. This says the comparison itself was void.
    probe_failure: str | None = None

    @property
    def ok(self) -> bool:
        return (
            self.skipped_reason is None
            and self.probe_failure is None
            and bool(self.results)
            and all(r.passed for r in self.results)
        )

    @property
    def failures(self) -> list[CaseResult]:
        return [r for r in self.results if not r.passed]

    @property
    def verdicts(self) -> dict[str, int]:
        """How many cases were better / equivalent / worse than the reference."""
        out: dict[str, int] = {}
        for r in self.results:
            if r.arbitration:
                out[r.arbitration.verdict] = out.get(r.arbitration.verdict, 0) + 1
        return out

    @property
    def accuracy_gain(self) -> float | None:
        """Largest factor by which the candidate beat the reference, if it did.

        The verdict counts say how often the candidate won; they do not say by
        how much. "better=84" and a 1.02x edge read identically, so the margin
        is reported alongside them and comes from the same measurements.
        """
        # Only cases the verdict actually called a win. The ratio is the tensor's
        # aggregate margin while the verdict is decided per row, so a case can
        # be closer overall and still be rejected for one row it got wrong.
        # Quoting that as a win would advertise a margin off a failed case.
        wins = [
            r.arbitration.ratio
            for r in self.results
            if r.arbitration
            and r.arbitration.verdict == "better"
            and 0.0 < r.arbitration.ratio < 1.0
        ]
        return 1.0 / min(wins) if wins else None

    @property
    def worst_ulp(self) -> int:
        return max((r.max_ulp or 0) for r in self.results) if self.results else 0

    @property
    def worst_abs(self) -> float | None:
        """Largest absolute divergence from the reference, in output units."""
        errs = [r.max_abs for r in self.results if r.max_abs is not None]
        return max(errs) if errs else None

    def as_dict(self) -> dict:
        """Everything the printed line says, and the per-case detail it elides.

        `__str__` shows at most the first eight failures because a broken kernel
        fails everything; this shows all of them, which is what a machine wants.
        """
        return {
            "name": self.name,
            "ok": self.ok,
            "skipped_reason": self.skipped_reason,
            "probe_failure": self.probe_failure,
            "cases": len(self.results),
            "passed": len(self.results) - len(self.failures),
            "verdicts": self.verdicts,
            "accuracy_gain": self.accuracy_gain,
            "worst_ulp": self.worst_ulp if self.results else None,
            "worst_abs": self.worst_abs,
            "results": [r.as_dict() for r in self.results],
        }

    def __str__(self) -> str:
        if self.skipped_reason:
            return f"SKIP  {self.name}: {self.skipped_reason}"
        if self.probe_failure:
            return f"FAIL  {self.name}: REFERENCE IS NOT SANE, {self.probe_failure}"
        head = (
            f"{'PASS' if self.ok else 'FAIL'}  {self.name}: "
            f"{len(self.results) - len(self.failures)}/{len(self.results)} cases, "
            f"worst ulp={self.worst_ulp}"
        )
        # A large ULP count on near-zero outputs is arithmetic, not a defect, so
        # the magnitude that produced it travels with it.
        if self.worst_abs is not None:
            head += f" (max abs {self.worst_abs:.3e})"
        # In oracle mode the ULP figure is expected to be large and says little
        # on its own. The accuracy verdict is the number that matters.
        if self.verdicts:
            parts = ", ".join(f"{v}={n}" for v, n in sorted(self.verdicts.items()))
            gain = self.accuracy_gain
            margin = f", up to {gain:.0f}x closer to float64" if gain and gain >= 1.5 else ""
            head += f"  [accuracy vs original: {parts}{margin}]"
        # Only the first few failures; a broken kernel fails everything.
        return "\n".join([head, *(str(f) for f in self.failures[:8])])


@dataclass
class Harness:
    """Run a candidate against a reference over a sweep of cases."""

    candidate: Callable[..., torch.Tensor]
    reference: Reference | Callable[..., torch.Tensor]
    name: str = "candidate"
    max_ulp: int = 4
    distributions: Sequence[Distribution] = DEFAULT_SWEEP
    determinism_runs: int = 3
    # "auto" puts inputs on the GPU when one is present. A Triton candidate
    # cannot accept CPU tensors ("Pointer argument cannot be accessed from
    # Triton"), while NativeReference copies to host regardless, so the device
    # is chosen for the candidate and everything is normalised for comparison.
    device: str = "auto"
    # A float64 implementation of the same maths. When supplied, correctness is
    # judged by accuracy against this oracle rather than by agreement with the
    # reference, because the reference is frequently the less accurate side.
    # See compare.arbitrate for the measurements behind this.
    oracle: Callable[..., torch.Tensor] | None = None
    oracle_slack: float = 2.0
    # Additional operands, derived from the primary case so a shape sweep sweeps
    # them too. Each entry maps the primary InputSpec to a spec for one more
    # tensor, handed to candidate, reference and oracle in order after it.
    # Empty by default: a single-input kernel behaves exactly as before.
    extras: Sequence[Callable[[InputSpec, str], Any]] = ()
    # Precisions to sweep. float32 alone by default, because that is what the
    # native driver reads and what every shipped example kernel declares.
    # Inference runs in half precision, so a candidate meant for it should be
    # swept in half precision, judged against the same float64 oracle.
    dtypes: Sequence[Any] = ()
    # How many cases to spend sanity-checking the reference before trusting any
    # comparison with it. Zero disables the probe, for a caller that has already
    # established its reference or is deliberately measuring a broken one.
    probe_cases: int = 6
    # Memory layouts to check beyond contiguous. Empty by default so bench and
    # any caller timing a single shape are unaffected; verify and port opt in.
    # Every kernel indexes as `row * row_stride + col * col_stride`, and until
    # those strides were passed the second factor was assumed to be 1, which
    # nothing could catch because generate() only ever built contiguous
    # tensors: the assumption and the test data agreed with each other.
    layouts: Sequence[Any] = ()

    def extra_inputs(self, spec: InputSpec) -> list[Any]:
        """Build every extra operand for this case, on the sweep's device."""
        return [make(spec, self.device) for make in self.extras]

    def __post_init__(self) -> None:
        if not isinstance(self.reference, Reference):
            self.reference = TorchReference(self.reference)
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _one(self, spec: InputSpec) -> CaseResult:
        label = spec.describe()
        primary = generate(spec, device=self.device)
        # Extra operands, for kernels that take more than the tensor being
        # transformed. LayerNorm's gamma and beta are the reason this exists: a
        # kernel that scales by learned weights cannot be proved by a harness
        # that only ever hands it one tensor, and an unproven substitution is
        # worth nothing. Their shapes derive from the primary case, so a sweep
        # over shapes sweeps the weights with it.
        extras = self.extra_inputs(spec)
        ins = (primary, *extras)
        ins_dev = ins

        try:
            got = self.candidate(*ins)
        except Exception as exc:  # noqa: BLE001 - a raising kernel is a result, not a crash
            why = f"candidate raised {type(exc).__name__}: {exc}"
            return CaseResult(label, False, failures=[why])

        try:
            want = self.reference(*ins)
        except Exception as exc:  # noqa: BLE001
            why = f"reference raised {type(exc).__name__}: {exc}"
            return CaseResult(label, False, failures=[why])

        # Comparison happens on the host. The candidate may return a device
        # tensor while the reference returns a host one, and mixing them raises
        # rather than comparing.
        got = got.detach().cpu()
        if torch.is_tensor(want):
            want = want.detach().cpu()
        ins = tuple(t.detach().cpu() for t in ins)
        x = ins[0]

        extra: list[str] = []

        # Determinism: identical input must give a bitwise identical answer.
        # NaN-aware, because torch.equal treats NaN as unequal to itself and a
        # kernel that reliably produces NaN is still deterministic.
        for _ in range(max(0, self.determinism_runs - 1)):
            again = self.candidate(*ins_dev).detach().cpu()
            if not _same_bits(again, got):
                extra.append("NONDETERMINISTIC: repeated run differed bitwise")
                break

        # Identity: the failure mode that survives every loose tolerance check.
        # Judged against the primary input only. A weight tensor is not what a
        # dropped computation would echo back.
        if is_identity(got, x) and not is_identity(want, x):
            extra.append("IDENTITY: output is a byte-exact copy of the input")

        report = check(label, lambda *_t: got, lambda *_t: want, ins, max_ulp=self.max_ulp)

        if self.oracle is None:
            failures = [f for f in report.failures if "IDENTITY" not in f] + extra
            return CaseResult(label, not failures, report.max_ulp, failures, max_abs=report.max_abs)

        # Oracle mode: ULP drift from the reference is expected and fine as long
        # as the candidate is not further from the truth than the reference is.
        truth = self.oracle(*(t.double() for t in ins))
        arb = arbitrate(got, want, truth, slack=self.oracle_slack)
        failures = [f for f in report.failures if "exceeds tolerance" not in f]
        failures = [f for f in failures if "IDENTITY" not in f] + extra

        # Ranking is relative, so it cannot fail a case both sides get wrong the
        # same way. This can. A candidate that returns NaN or an infinity where
        # the oracle returns a number is not a less accurate implementation, it
        # is a broken one, and it stays broken when the reference agrees.
        broken = nonfinite_where_finite(got, truth)
        if broken:
            failures.append(f"NOT FINITE: {broken} position(s) where the oracle is a number")
        if arb.verdict == "worse":
            failures.append(f"LESS ACCURATE than the reference: {arb}")
        return CaseResult(label, not failures, report.max_ulp, failures, arb, report.max_abs)

    def _probe_specs(self, shapes, sweep):
        """A handful of cases spanning the sweep rather than one corner of it.

        The probe this replaces ran one shape, (4, 256), one distribution, in
        float32, which leaves three ways for a broken reference to pass it. A
        kernel with a fixed-size `__shared__ float s[256]` is correct at 256
        columns and garbage at 4096. A kernel that only misbehaves in half
        precision is never asked. And the widest and tallest shapes, where block
        and grid mistakes live, are not sampled at all.
        """
        picks: list[tuple[int, ...]] = []
        for shape in (shapes[0], max(shapes, key=lambda s: s[-1]), max(shapes, key=lambda s: s[0])):
            if tuple(shape) not in picks:
                picks.append(tuple(shape))

        dists = list(self.distributions) or [Distribution.NORMAL]
        chosen = [dists[0]] if len(dists) == 1 else [dists[0], dists[-1]]

        out = []
        for shape in picks:
            for dist in chosen:
                for dtype in sweep:
                    out.append(InputSpec(shape=shape, dtype=dtype, distribution=dist, seed=9973))
        return out[: self.probe_cases]

    def _probe_reference(self, shapes, sweep) -> str | None:
        """Refuse a reference that cannot reproduce its own maths.

        Oracle mode passes the candidate when it is closer to the truth than the
        reference is. That is the right rule and it has a hole: if the reference
        is garbage, the candidate is trivially closer and the run reports a
        proof. Seen for real, with a tuned kernel launched at the wrong block
        size, which read uninitialised shared memory and scored the candidate
        3.9e75x better.

        This lived in `port` alone, so `verify`, `bench` and `synth` ran without
        it. `synth` is the one that executes unreviewed generated code and whose
        docstring calls the gate its entire value.

        The threshold is relative and follows the precision. An absolute 1e-3
        was meaningless for a 256-wide softmax whose outputs are around 4e-3,
        where a reference 20% wrong still read as sane, and it would fail
        bfloat16 for being bfloat16. Sanity is a loose question: what this
        catches is orders of magnitude out, not fractions of a ULP.
        """
        for spec in self._probe_specs(shapes, sweep):
            ins = [generate(spec, device=self.device), *self.extra_inputs(spec)]
            try:
                got = self.reference(*ins)
            except Exception as exc:  # noqa: BLE001 - reported, never swallowed
                # A compile error, a launch fault, a kernel selected that is not
                # __global__: every one of them arrived as a raw traceback
                # before, out of a call nobody had wrapped.
                return (
                    f"the reference could not be run on {spec.describe()}: "
                    f"{type(exc).__name__}: {exc}"
                )

            truth = self.oracle(*(t.double() for t in ins))
            err = float((got.double().cpu() - truth.cpu()).abs().max())
            scale = float(truth.abs().max())
            tolerance = max(1e-3, 32.0 * torch.finfo(spec.dtype).eps)
            relative = err / scale if scale > 0 else err
            if not (relative <= tolerance):
                return (
                    f"disagrees with the float64 oracle by {relative:.3e} relative "
                    f"on {spec.describe()}, tolerance {tolerance:.3e}. It does not "
                    f"compute what was proposed, or it was launched wrongly, and "
                    f"being closer to the truth than a broken baseline is not "
                    f"evidence of anything."
                )
        return None

    def run(self, shapes: Iterable[tuple[int, ...]], limit: int | None = None) -> Summary:
        if isinstance(self.reference, Reference):
            avail = self.reference.availability()
            if not avail:
                return Summary(self.name, skipped_reason=avail.reason)

        shapes = [tuple(s) for s in shapes]
        summary = Summary(self.name)
        n = 0
        sweep = self.dtypes or (torch.float32,)

        # Before anything is ranked against the reference, establish that the
        # reference is worth ranking against. Only meaningful in oracle mode:
        # with no truth to check it against there is nothing to check.
        if self.oracle is not None and self.probe_cases and shapes:
            summary.probe_failure = self._probe_reference(shapes, sweep)
            if summary.probe_failure:
                return summary

        specs = [
            InputSpec(shape=shape, dtype=dtype, distribution=dist, seed=i)
            for i, (shape, dist, dtype) in enumerate(
                (sh, di, dt) for sh in shapes for di in self.distributions for dt in sweep
            )
        ]
        if limit is not None:
            specs = specs[:limit]

        # A short layout pass, appended rather than crossed with everything.
        # Crossing four layouts with seven distributions and three precisions
        # would quadruple a metered run; this adds one case per layout per
        # precision, at the widest shape, which is where a stride mistake shows.
        #
        # It sits after the truncation on purpose. --limit is a cost control,
        # and letting it silently drop the only cases that exercise strides
        # would repeat the defect the multi-row sweep fix was written for.
        if self.layouts and shapes:
            widest = max(shapes, key=lambda sh: sh[-1])
            base = self.distributions[0] if self.distributions else Distribution.NORMAL
            specs += [
                InputSpec(
                    shape=widest,
                    dtype=dtype,
                    distribution=base,
                    seed=len(specs) + i,
                    layout=layout,
                )
                for i, (layout, dtype) in enumerate((la, dt) for la in self.layouts for dt in sweep)
            ]

        for spec in specs:
            summary.results.append(self._one(spec))
            n += 1
        return summary


__all__ = ["CaseResult", "Harness", "Summary"]

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

import torch

from hipbridge.verify.compare import Arbitration, arbitrate, check, is_identity
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

    @property
    def ok(self) -> bool:
        return (
            self.skipped_reason is None
            and bool(self.results)
            and all(r.passed for r in self.results)
        )

    @property
    def failures(self) -> list[CaseResult]:
        return [r for r in self.results if not r.passed]

    @property
    def worst_ulp(self) -> int:
        return max((r.max_ulp or 0) for r in self.results) if self.results else 0

    def __str__(self) -> str:
        if self.skipped_reason:
            return f"SKIP  {self.name}: {self.skipped_reason}"
        head = (
            f"{'PASS' if self.ok else 'FAIL'}  {self.name}: "
            f"{len(self.results) - len(self.failures)}/{len(self.results)} cases, "
            f"worst ulp={self.worst_ulp}"
        )
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
    device: str = "cpu"
    # A float64 implementation of the same maths. When supplied, correctness is
    # judged by accuracy against this oracle rather than by agreement with the
    # reference, because the reference is frequently the less accurate side.
    # See compare.arbitrate for the measurements behind this.
    oracle: Callable[..., torch.Tensor] | None = None
    oracle_slack: float = 2.0

    def __post_init__(self) -> None:
        if not isinstance(self.reference, Reference):
            self.reference = TorchReference(self.reference)

    def _one(self, spec: InputSpec) -> CaseResult:
        label = spec.describe()
        x = generate(spec, device=self.device)

        try:
            got = self.candidate(x)
        except Exception as exc:  # noqa: BLE001 - a raising kernel is a result, not a crash
            why = f"candidate raised {type(exc).__name__}: {exc}"
            return CaseResult(label, False, failures=[why])

        try:
            want = self.reference(x)
        except Exception as exc:  # noqa: BLE001
            why = f"reference raised {type(exc).__name__}: {exc}"
            return CaseResult(label, False, failures=[why])

        extra: list[str] = []

        # Determinism: identical input must give a bitwise identical answer.
        # NaN-aware, because torch.equal treats NaN as unequal to itself and a
        # kernel that reliably produces NaN is still deterministic.
        for _ in range(max(0, self.determinism_runs - 1)):
            again = self.candidate(x)
            if not _same_bits(again, got):
                extra.append("NONDETERMINISTIC: repeated run differed bitwise")
                break

        # Identity: the failure mode that survives every loose tolerance check.
        if is_identity(got, x) and not is_identity(want, x):
            extra.append("IDENTITY: output is a byte-exact copy of the input")

        report = check(label, lambda _t: got, lambda _t: want, (x,), max_ulp=self.max_ulp)

        if self.oracle is None:
            failures = [f for f in report.failures if "IDENTITY" not in f] + extra
            return CaseResult(label, not failures, report.max_ulp, failures)

        # Oracle mode: ULP drift from the reference is expected and fine as long
        # as the candidate is not further from the truth than the reference is.
        arb = arbitrate(got, want, self.oracle(x.double()), slack=self.oracle_slack)
        failures = [f for f in report.failures if "exceeds tolerance" not in f]
        failures = [f for f in failures if "IDENTITY" not in f] + extra
        if arb.verdict == "worse":
            failures.append(f"LESS ACCURATE than the reference: {arb}")
        return CaseResult(label, not failures, report.max_ulp, failures, arb)

    def run(self, shapes: Iterable[tuple[int, ...]], limit: int | None = None) -> Summary:
        if isinstance(self.reference, Reference):
            avail = self.reference.availability()
            if not avail:
                return Summary(self.name, skipped_reason=avail.reason)

        summary = Summary(self.name)
        n = 0
        for shape in shapes:
            for dist in self.distributions:
                if limit is not None and n >= limit:
                    return summary
                summary.results.append(
                    self._one(InputSpec(shape=tuple(shape), distribution=dist, seed=n))
                )
                n += 1
        return summary


__all__ = ["CaseResult", "Harness", "Summary"]

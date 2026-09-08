"""Device-side benchmarking of a candidate against one or more originals.

Correctness alone does not justify substituting a kernel. If the tuned
implementation is not faster there is no reason to use it, so this measures
every side on the same device, at the same shape, with the same methodology.

Method, stated because a benchmark without one is an anecdote:

  - each native kernel is timed inside the generated driver with device events,
    so file I/O and host copies are excluded
  - the candidate is timed with device events too, never wall clock
  - every side gets warmup iterations before any timing, to pay JIT and cache
    costs once rather than charge them to the first measurement
  - one run is the mean over `reps` back-to-back launches; the reported figure
    is the MEDIAN across `runs` such runs, with the min and max beside it,
    because a single mean hides how much small shapes move between runs
  - a shape is timed only after it has been verified correct at that shape;
    a fast wrong kernel is not a result

Two baselines rather than one. The naive `row_softmax.cu` launches one thread
per block, so beating it proves little: most of that ratio is its launch
configuration, not the language it is written in. `row_softmax_tuned.cu` is the
kernel a competent engineer would write for the same hardware, and the honest
question is what the substitution buys against THAT. A second ratio near 1.0 is
a real finding and worth more than a large number against a strawman.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from hipbridge.verify.reference import NativeReference

# A shape is called latency-bound when its throughput falls this far below the
# best throughput the same implementation reached at any shape in the sweep.
#
# Why a quarter. Throughput here is elements per millisecond, so a shape at
# fraction f of the best is spending (1 - f) of its time on something other than
# the work. At 0.25 that is three quarters of the measurement, which is past any
# reading of "the kernel's speed" and safely clear of ordinary variation: the
# same shape moves by about 15% between runs on the MI300X, and the observed
# small shapes sit near 0.001 of the best rather than anywhere near the line.
# The threshold is therefore not delicately placed, and the label is a warning
# about interpretation rather than a precise classification.
#
# An absolute rule was considered and rejected: compare each time against the
# measured cost of launching a kernel that does no work (4.9 us from Python on
# that box). It fails because each implementation has its own launch cost, and
# Triton's is roughly 17 us against torch's 5.5 us, so one absolute threshold
# would label the same shape differently depending on who is being timed. The
# relative rule asks the only question that survives that: is this shape slow
# compared with how fast this same implementation can go.
LATENCY_BOUND_FRACTION = 0.25

# Below this many shapes there is nothing to compare against, so no shape is
# labelled at all. A single-shape sweep is its own best throughput by
# definition, and calling it compute-bound on that basis would be circular.
LATENCY_BOUND_MIN_SHAPES = 2


@dataclass(frozen=True)
class Stat:
    """Per-iteration milliseconds across repeated timing runs."""

    runs: tuple[float, ...]

    @property
    def median(self) -> float:
        return statistics.median(self.runs)

    @property
    def lo(self) -> float:
        return min(self.runs)

    @property
    def hi(self) -> float:
        return max(self.runs)

    @property
    def spread(self) -> float:
        """Max over min: how far the same measurement moved between runs."""
        return self.hi / self.lo if self.lo else float("inf")

    def us(self) -> str:
        if len(self.runs) == 1:
            return f"{self.median * 1000:.1f}"
        return f"{self.median * 1000:.1f} [{self.lo * 1000:.1f}-{self.hi * 1000:.1f}]"


@dataclass(frozen=True)
class Measurement:
    """One implementation, timed at one shape."""

    name: str
    stat: Stat
    device: str = "unknown"

    def as_dict(self) -> dict:
        """Microseconds, which is what the table prints and what a reader compares.

        The spread travels with the median rather than being summarised away: a
        median that sits outside a wide min-max band is a measurement someone
        should look at again before quoting it.
        """
        return {
            "name": self.name,
            "device": self.device,
            "median_us": self.stat.median * 1000.0,
            "min_us": self.stat.lo * 1000.0,
            "max_us": self.stat.hi * 1000.0,
            "runs": len(self.stat.runs),
        }


@dataclass
class ShapeRow:
    """Every implementation timed at a single shape."""

    shape: tuple[int, ...]
    candidate: Measurement
    baselines: tuple[Measurement, ...] = ()
    verified: bool = False
    latency_bound: bool = False
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        """A row of the printed table, with the ratios it computes for itself.

        `verified` and `comparable` are the two facts that decide whether any of
        these numbers may be quoted, so they are fields rather than something a
        consumer has to reconstruct: a shape that failed verification is timed
        but not a result, and a candidate that ran on the host is not comparable
        to a baseline that ran on device.
        """
        return {
            "shape": list(self.shape),
            "elements": self.elements,
            "verified": self.verified,
            "comparable": self.comparable,
            "latency_bound": self.latency_bound,
            "notes": list(self.notes),
            "candidate": self.candidate.as_dict(),
            "baselines": [b.as_dict() for b in self.baselines],
            "speedup": {b.name: self.speedup(b.name) for b in self.baselines},
        }

    @property
    def comparable(self) -> bool:
        """Were the candidate and the baselines timed on the same kind of part?

        Native baselines always run on device. If the candidate ran on the host,
        the ratio compares a CPU implementation against a GPU one and means
        nothing. Reporting a speedup for that is how misleading benchmark tables
        get made, so it is refused rather than footnoted.
        """
        return self.candidate.device == "cuda"

    @property
    def elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    def speedup(self, name: str) -> float | None:
        if not self.comparable:
            return None
        for b in self.baselines:
            if b.name == name:
                c = self.candidate.stat.median
                return b.stat.median / c if c else float("inf")
        return None


def mark_latency_bound(rows: list[ShapeRow]) -> bool:
    """Flag shapes where the candidate never gets to show its throughput.

    Compares each shape's elements-per-millisecond against the best the same
    candidate managed anywhere in the sweep. Well under that best means the
    kernel spent its time being launched rather than working, so the ratio there
    is a statement about dispatch overhead and should not be read as throughput.

    Returns whether the sweep was large enough to judge. A short sweep leaves
    every row unlabelled rather than quietly declaring them all compute-bound,
    which is the answer a single shape would always give about itself.
    """
    rates = [(r, r.elements / r.candidate.stat.median) for r in rows if r.candidate.stat.median]
    if len({r.shape for r, _ in rates}) < LATENCY_BOUND_MIN_SHAPES:
        for row, _ in rates:
            row.latency_bound = False
        return False

    best = max(rate for _, rate in rates)
    for row, rate in rates:
        row.latency_bound = rate < best * LATENCY_BOUND_FRACTION
    return True


def render(rows: list[ShapeRow]) -> str:
    """One aligned table for the whole sweep."""
    if not rows:
        return "no shapes measured"
    judged = mark_latency_bound(rows)
    names = [b.name for b in rows[0].baselines]

    head = f"{'shape':>12}  " + "".join(f"{n + ' (us)':>24}" for n in names)
    head += f"{'candidate (us)':>24}" + "".join(f"{'vs ' + n:>13}" for n in names)
    lines = [head, "-" * len(head)]

    for r in rows:
        dims = "x".join(str(d) for d in r.shape)
        line = f"{dims:>12}  " + "".join(f"{b.stat.us():>24}" for b in r.baselines)
        line += f"{r.candidate.stat.us():>24}"
        if r.comparable:
            for n in names:
                s = r.speedup(n)
                line += f"{s:>12.1f}x" if s is not None else f"{'-':>13}"
        else:
            line += f"  NOT COMPARABLE: candidate ran on {r.candidate.device}, not the device"
        flags = []
        if not r.verified:
            flags.append("UNVERIFIED at this shape")
        if r.latency_bound and r.comparable:
            flags.append("latency-bound")
        flags += r.notes
        if flags:
            line += "  (" + "; ".join(flags) + ")"
        lines.append(line)
    if not judged:
        lines.append(
            f"  (fewer than {LATENCY_BOUND_MIN_SHAPES} distinct shapes, so nothing is "
            "labelled latency-bound: there is no faster run of the same kernel to "
            "compare against)"
        )
    return "\n".join(lines)


def _as_tuple(ins) -> tuple[torch.Tensor, ...]:
    """Accept one tensor or a tuple of them, so single-input callers are unchanged."""
    return (ins,) if torch.is_tensor(ins) else tuple(ins)


def time_candidate(
    fn: Callable[..., torch.Tensor],
    ins,
    reps: int = 100,
) -> float:
    """Per-iteration milliseconds, timed with device events.

    `ins` is the full operand list. A kernel taking weights has to be timed with
    the weights it was proved with, or the measurement describes a different
    program from the one that passed verification.
    """
    ins = _as_tuple(ins)
    x = ins[0]
    warmup = max(3, reps // 10)
    for _ in range(warmup):
        fn(*ins)

    if x.is_cuda:
        torch.cuda.synchronize()
        beg, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        beg.record()
        for _ in range(reps):
            fn(*ins)
        end.record()
        torch.cuda.synchronize()
        return beg.elapsed_time(end) / reps

    # CPU fallback: still measured, just not comparable to device numbers.
    import time

    t0 = time.perf_counter()
    for _ in range(reps):
        fn(*ins)
    return (time.perf_counter() - t0) * 1000.0 / reps


def time_reference(ref: NativeReference, ins, reps: int = 100) -> float:
    """Per-iteration milliseconds for a native kernel, timed on device."""
    ref._reps = reps
    try:
        ref(*_as_tuple(ins))
    finally:
        ref._reps = 0
    if ref.last_kernel_ms is None:
        raise RuntimeError(
            "the driver reported no KERNEL_MS line; timing is unavailable for this build"
        )
    return ref.last_kernel_ms


def stat_candidate(fn: Callable[..., torch.Tensor], ins, reps: int = 100, runs: int = 5) -> Stat:
    """Repeat the candidate timing `runs` times and keep every result."""
    return Stat(tuple(time_candidate(fn, ins, reps) for _ in range(max(1, runs))))


def stat_reference(ref: NativeReference, ins, reps: int = 100, runs: int = 5) -> Stat:
    """Repeat a native timing `runs` times and keep every result."""
    return Stat(tuple(time_reference(ref, ins, reps) for _ in range(max(1, runs))))


__all__ = [
    "LATENCY_BOUND_FRACTION",
    "LATENCY_BOUND_MIN_SHAPES",
    "Measurement",
    "ShapeRow",
    "Stat",
    "mark_latency_bound",
    "render",
    "stat_candidate",
    "stat_reference",
    "time_candidate",
    "time_reference",
]

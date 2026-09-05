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
# Below that, the measurement is dominated by fixed launch overhead and the
# ratio says more about dispatch than about the kernel.
LATENCY_BOUND_FRACTION = 0.25


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


@dataclass
class ShapeRow:
    """Every implementation timed at a single shape."""

    shape: tuple[int, ...]
    candidate: Measurement
    baselines: tuple[Measurement, ...] = ()
    verified: bool = False
    latency_bound: bool = False
    notes: list[str] = field(default_factory=list)

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


def mark_latency_bound(rows: list[ShapeRow]) -> None:
    """Flag shapes where the candidate never gets to show its throughput.

    Compares each shape's elements-per-millisecond against the best the same
    candidate managed anywhere in the sweep. Well under that best means the
    kernel spent its time being launched rather than working, so the ratio there
    is a statement about dispatch overhead and should not be read as throughput.
    """
    rates = [(r, r.elements / r.candidate.stat.median) for r in rows if r.candidate.stat.median]
    if not rates:
        return
    best = max(rate for _, rate in rates)
    for row, rate in rates:
        row.latency_bound = rate < best * LATENCY_BOUND_FRACTION


def render(rows: list[ShapeRow]) -> str:
    """One aligned table for the whole sweep."""
    if not rows:
        return "no shapes measured"
    mark_latency_bound(rows)
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
    return "\n".join(lines)


def time_candidate(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    reps: int = 100,
) -> float:
    """Per-iteration milliseconds, timed with device events."""
    warmup = max(3, reps // 10)
    for _ in range(warmup):
        fn(x)

    if x.is_cuda:
        torch.cuda.synchronize()
        beg, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        beg.record()
        for _ in range(reps):
            fn(x)
        end.record()
        torch.cuda.synchronize()
        return beg.elapsed_time(end) / reps

    # CPU fallback: still measured, just not comparable to device numbers.
    import time

    t0 = time.perf_counter()
    for _ in range(reps):
        fn(x)
    return (time.perf_counter() - t0) * 1000.0 / reps


def time_reference(ref: NativeReference, x: torch.Tensor, reps: int = 100) -> float:
    """Per-iteration milliseconds for a native kernel, timed on device."""
    ref._reps = reps
    try:
        ref(x)
    finally:
        ref._reps = 0
    if ref.last_kernel_ms is None:
        raise RuntimeError(
            "the driver reported no KERNEL_MS line; timing is unavailable for this build"
        )
    return ref.last_kernel_ms


def stat_candidate(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    reps: int = 100,
    runs: int = 5,
) -> Stat:
    """Repeat the candidate timing `runs` times and keep every result."""
    return Stat(tuple(time_candidate(fn, x, reps) for _ in range(max(1, runs))))


def stat_reference(ref: NativeReference, x: torch.Tensor, reps: int = 100, runs: int = 5) -> Stat:
    """Repeat a native timing `runs` times and keep every result."""
    return Stat(tuple(time_reference(ref, x, reps) for _ in range(max(1, runs))))


__all__ = [
    "LATENCY_BOUND_FRACTION",
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

"""Device-side benchmarking of a candidate against the original kernel.

Correctness alone does not justify substituting a kernel. If the tuned
implementation is not faster there is no reason to use it, so this measures
both sides on the same device, at the same shape, with the same methodology.

Method, stated because a benchmark without one is an anecdote:

  - the original is timed inside the generated driver with device events, so
    file I/O and host copies are excluded
  - the candidate is timed with device events too, never wall clock
  - both get warmup iterations before any timing, to pay JIT and cache costs
  - the reported figure is per-iteration mean over `reps` back-to-back launches
  - a shape is timed only after it has been verified correct at that shape;
    a fast wrong kernel is not a result
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from hipbridge.verify.reference import NativeReference


@dataclass
class BenchResult:
    name: str
    ms: float
    reps: int
    shape: tuple[int, ...]
    device: str = "unknown"

    def __str__(self) -> str:
        return f"{self.name} on {self.device}: {self.ms * 1000:.1f} us/iter over {self.reps} reps"


@dataclass
class Comparison:
    candidate: BenchResult
    reference: BenchResult
    verified: bool

    @property
    def comparable(self) -> bool:
        """Were both sides timed on the same kind of hardware?

        The reference always runs on device. If the candidate ran on the host,
        the ratio compares a CPU implementation against a GPU one and means
        nothing. Reporting a speedup for that is how misleading benchmark tables
        get made, so it is refused rather than footnoted.
        """
        return self.candidate.device == "cuda"

    @property
    def speedup(self) -> float:
        return self.reference.ms / self.candidate.ms if self.candidate.ms else float("inf")

    def __str__(self) -> str:
        dims = "x".join(str(d) for d in self.candidate.shape)
        if not self.comparable:
            return (
                f"{dims:>12}  original {self.reference.ms * 1000:9.1f} us (device)   "
                f"candidate {self.candidate.ms * 1000:9.1f} us ({self.candidate.device})   "
                f"NOT COMPARABLE: candidate did not run on the device"
            )
        flag = "" if self.verified else "  (UNVERIFIED at this shape)"
        return (
            f"{dims:>12}  original {self.reference.ms * 1000:9.1f} us   "
            f"candidate {self.candidate.ms * 1000:9.1f} us   "
            f"{self.speedup:6.1f}x{flag}"
        )


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
    """Per-iteration milliseconds for the original kernel, timed on device."""
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


def compare(
    candidate: Callable[[torch.Tensor], torch.Tensor],
    reference: NativeReference,
    x: torch.Tensor,
    reps: int = 100,
    verified: bool = False,
) -> Comparison:
    shape = tuple(x.shape)
    dev = "cuda" if x.is_cuda else "cpu"
    return Comparison(
        candidate=BenchResult("candidate", time_candidate(candidate, x, reps), reps, shape, dev),
        reference=BenchResult("original", time_reference(reference, x, reps), reps, shape, "cuda"),
        verified=verified,
    )


__all__ = ["BenchResult", "Comparison", "compare", "time_candidate", "time_reference"]

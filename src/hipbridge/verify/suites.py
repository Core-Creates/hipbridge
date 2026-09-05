"""Verification suites for the shipped example kernels.

A suite ties together the four things needed to prove a substitution:

  source      the original .cu, compiled and run on device as ground truth
  launch      how to launch it (grid, block, scalar arguments)
  candidate   the implementation we propose to substitute in its place
  oracle      the same maths in float64, to arbitrate when the two disagree

The example kernels compile under hipcc unchanged because they use only
constructs HIP spells identically: __global__, threadIdx, blockIdx, __shared__,
__syncthreads, fmaxf, expf. A kernel touching cudaMalloc or cuBLAS would need
hipifying first; these do not.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from hipbridge.verify.reference import LaunchSpec


@dataclass(frozen=True)
class Baseline:
    """A native kernel to time the candidate against.

    More than one, because a single baseline can flatter the candidate. The
    original as written answers "what does replacing this buy me"; a competently
    written version of the same maths answers "is the substitution worth making
    at all", which is the question a reviewer will actually ask.
    """

    name: str
    source_file: str
    launch: LaunchSpec

    def source(self, examples_dir: Path) -> str:
        return (examples_dir / self.source_file).read_text(encoding="utf-8")


@dataclass(frozen=True)
class Suite:
    name: str
    source_file: str
    kernel: str
    launch: LaunchSpec
    oracle: Callable[[torch.Tensor], torch.Tensor]
    shapes: Callable[[], Sequence[tuple[int, ...]]]
    baselines: tuple[Baseline, ...] = ()
    # The library call a user would reach for instead of substituting anything.
    # Timed on the same device as the candidate, because "why not just use
    # torch" is the first question anyone sensible asks about a substitution.
    portable: Callable[[torch.Tensor], torch.Tensor] | None = None
    portable_name: str = "torch"

    def source(self, examples_dir: Path) -> str:
        return (examples_dir / self.source_file).read_text(encoding="utf-8")

    def all_baselines(self) -> tuple[Baseline, ...]:
        """The original first, then any additional baselines."""
        return (Baseline("original", self.source_file, self.launch), *self.baselines)


def _row_shapes() -> Sequence[tuple[int, ...]]:
    from hipbridge.verify import shapes

    return shapes.sample(shapes.row_wise())


ROW_SOFTMAX = Suite(
    name="row_softmax",
    source_file="row_softmax.cu",
    kernel="row_softmax",
    launch=LaunchSpec(
        kernel="row_softmax",
        grid=lambda s: (s[0][0], 1, 1),  # one block per row
        block=(1, 1, 1),  # serial within the row, as written
        scalar_args=lambda s: [s[0][0], s[0][1]],
        out_shape=lambda s: s[0],
    ),
    oracle=lambda t: torch.softmax(t.double(), dim=-1),
    portable=lambda t: torch.softmax(t, dim=-1),
    shapes=_row_shapes,
    baselines=(
        Baseline(
            name="tuned HIP",
            source_file="row_softmax_tuned.cu",
            launch=LaunchSpec(
                kernel="row_softmax_tuned",
                grid=lambda s: (s[0][0], 1, 1),  # one block per row
                block=(256, 1, 1),  # four wavefronts, tree-reduced in LDS
                scalar_args=lambda s: [s[0][0], s[0][1]],
                out_shape=lambda s: s[0],
            ),
        ),
    ),
)

BUILTIN: tuple[Suite, ...] = (ROW_SOFTMAX,)


def candidate_for(suite: Suite) -> tuple[Callable[[torch.Tensor], torch.Tensor], str]:
    """The implementation being proposed in place of the original.

    Prefers the tuned AMD Triton kernel when the [kernels] extra is installed,
    which is the substitution this project actually exists to make. Falls back to
    a torch implementation so the suite still runs and still means something on a
    machine without Triton.
    """
    if suite.name == "row_softmax":
        from hipbridge import kernels

        if kernels.available():
            from hipbridge.kernels.softmax import softmax_rowwise

            return softmax_rowwise, "hipbridge.kernels.softmax (Triton, AMD-tuned)"
        return (lambda t: torch.softmax(t, dim=-1)), "torch.softmax (Triton unavailable)"
    raise KeyError(suite.name)


__all__ = ["BUILTIN", "ROW_SOFTMAX", "Baseline", "Suite", "candidate_for"]

"""Elementwise recognizer.

Deliberately the strictest rule in the set, and deliberately last. Elementwise is
the shape everything degrades into when a classifier gives up, so it must be
claimed only on positive evidence: no loops, no shared memory, no barriers, no
cross-lane communication.
"""

from __future__ import annotations

from drover.frontend.ir import KernelFacts, Pattern
from drover.recognize.base import rule


@rule("elementwise.flat", priority=90)
def flat_elementwise(f: KernelFacts):
    disqualifying = (
        f.loops,
        len(f.shared),
        f.barriers,
        len(f.shuffle_intrinsics),
        len(f.atomics),
        f.shared_accumulations,
    )
    if any(disqualifying):
        return None
    if not (f.uses_thread_index and f.outputs):
        return None
    return (
        Pattern.ELEMENTWISE,
        "certain",
        [
            "no loops, no shared memory, no barriers, no cross-lane ops",
            f"{len(f.inputs)} input pointer(s), {len(f.outputs)} output pointer(s)",
            "one output element per thread",
        ],
    )

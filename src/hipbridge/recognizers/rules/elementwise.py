"""Elementwise recognizer.

Deliberately the strictest rule in the set, and deliberately last. Elementwise is
the shape everything degrades into when a classifier gives up, so it must be
claimed only on positive evidence: no loops, no shared memory, no barriers, no
cross-lane communication.
"""

from __future__ import annotations

from hipbridge.frontend.ir import KernelFacts, Pattern
from hipbridge.recognizers.base import rule


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


@rule("elementwise.row_map", priority=95)
def row_map(f: KernelFacts):
    """A per-row map: walks the row, accumulates nothing.

    Sits between reduce.serial and elementwise.flat, and is separated from both
    by what it does NOT do. reduce.serial requires an accumulation across the
    row; this rule requires their absence, which is exactly what makes the row's
    outputs independent of each other. elementwise.flat requires no loop at all.

    Deliberately "likely" rather than "certain". The absence of an accumulator
    is good evidence that nothing is reduced, and it is not proof that the
    kernel does anything in particular, which is what the numeric proof is for.
    """
    disqualifying = (
        len(f.shared),
        f.barriers,
        len(f.shuffle_intrinsics),
        len(f.atomics),
        f.shared_accumulations,
        f.scalar_accumulations,
    )
    if any(disqualifying):
        return None
    if not (f.loops >= 1 and f.uses_block_index and f.outputs):
        return None
    return (
        Pattern.ROW_MAP,
        "likely",
        [
            f"{f.loops} loop(s) and no accumulation of any kind",
            "one row per block, so each row's outputs are independent",
            f"{len(f.inputs)} input pointer(s), {len(f.outputs)} output pointer(s)",
        ],
    )

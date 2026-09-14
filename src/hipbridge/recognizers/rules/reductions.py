"""Reduction recognizers.

Three distinct shapes that all reduce, and that lower to very different AMD code:

  tree     shared-memory log-depth tree, retuned to the target's wavefront
  shuffle  warp intrinsics, needs mask removal and wavefront-wide lane semantics
  serial   per-thread sequential accumulation, usually one row per block

Recognition runs before anyone has named a device, so the notes below cannot
know the width and must not imply one. They used to say "64 wide", which is
CDNA's answer and wrong for every Radeon: RDNA compiles 32 wide, where a tree
reduction over one wavefront takes 5 steps, not 6.
"""

from __future__ import annotations

from hipbridge.frontend.ir import KernelFacts, Pattern
from hipbridge.recognizers.base import rule


@rule("reduce.tree", priority=10)
def tree_reduction(f: KernelFacts):
    # A tree reduction is shared memory, barriers, and a halving stride that
    # touches that shared memory. It does NOT have to combine with `+=`: an
    # online softmax rescales a running sum onto a new maximum, and Welford
    # merges partial moments, both spelled as plain assignments. Requiring a
    # compound assignment made both invisible, and they are the competent
    # implementations, which is precisely the wrong thing to miss.
    combines = f.shared_accumulations or f.shared_in_halving_loop
    if not (f.shared and f.barriers >= 2 and f.has_halving_stride and combines):
        return None
    return (
        Pattern.REDUCE_TREE,
        "certain",
        [
            f"{len(f.shared)} shared buffer(s): "
            + ", ".join(
                f"{s.type}" + (f" ({s.size_bytes} B)" if s.size_bytes else "") for s in f.shared
            ),
            f"{f.barriers} __syncthreads() calls",
            "loop stride is halved (log-depth tree)",
            (
                f"{f.shared_accumulations} compound assignment(s) into shared memory"
                if f.shared_accumulations
                else "the halving loop combines values held in shared memory"
            ),
            "AMD note: tree depth follows the wavefront: 6 steps on CDNA (64 wide), "
            "5 on RDNA (32 wide)",
        ],
    )


@rule("reduce.shuffle", priority=20)
def shuffle_reduction(f: KernelFacts):
    if not f.shuffle_intrinsics:
        return None
    return (
        Pattern.REDUCE_SHUFFLE,
        "certain",
        [
            "warp shuffle intrinsics: " + ", ".join(f.shuffle_intrinsics),
            "AMD note: drop the mask argument, __shfl_down_sync -> __shfl_down",
            "AMD note: the wavefront width (64 on CDNA, 32 on RDNA) sets the reduction "
            "depth and ballot width",
        ],
    )


@rule("reduce.serial", priority=30)
def serial_reduction(f: KernelFacts):
    # One block per row, thread walks the row accumulating into a scalar.
    if not (f.loops >= 1 and f.scalar_accumulations >= 1 and not f.shared):
        return None
    if not f.uses_block_index:
        return None
    return (
        Pattern.REDUCE_SERIAL,
        "likely",
        [
            f"{f.loops} loop(s) with {f.scalar_accumulations} scalar accumulation(s)",
            "no shared memory, indexed by blockIdx (one row per block)",
            "candidate for substitution with a tuned row-wise Triton kernel",
        ],
    )

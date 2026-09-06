"""Deciding whether a recognized kernel has a substitute worth proposing.

A pattern is not a licence to substitute. `row_softmax.cu` is recognized as
`reduce_serial`, and so is a kernel that sums a row, and so is one that takes a
product. Swapping a softmax into any of them would be a catastrophe that the
recognizer alone cannot prevent, because the pattern describes the *shape* of
the computation, not the maths.

So a proposal needs two things the recognizer does not supply:

  1. corroborating evidence in the source, spelled out per substitution below,
     narrow enough that the guess is defensible
  2. a numeric proof, which is the part that actually makes it safe

Only the second one is load bearing. The evidence decides what to try; the
harness decides whether it was right, by compiling the caller's own kernel and
comparing both against a float64 oracle on device. A wrong guess fails there and
is reported as a failure, never as a substitution.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from hipbridge.frontend.ir import KernelFacts, Pattern
from hipbridge.verify.suites import ROW_SOFTMAX, Suite


@dataclass(frozen=True)
class Proposal:
    """A substitute worth trying, and why it was proposed."""

    suite: Suite
    evidence: list[str]

    @property
    def name(self) -> str:
        return self.suite.name


def _looks_like_softmax(source: str, facts: KernelFacts) -> list[str] | None:
    """Softmax has a signature no other row-wise reduction shares.

    Exponentiation, a maximum for numerical stability, and a division by the
    accumulated total. A row sum has the third and not the first two. Requiring
    all three is what keeps this from claiming every reduction it meets.
    """
    evidence = []
    if not re.search(r"\bexpf?\s*\(", source):
        return None
    evidence.append("calls expf, so it is not a plain sum")
    if not re.search(r"\bfmaxf?\s*\(|\bmax\s*\(", source):
        return None
    evidence.append("takes a maximum, the usual stability pass")
    if not re.search(r"/=|/\s*\w*(sum|total|s)\b", source):
        return None
    evidence.append("divides by an accumulated total")

    if len(facts.inputs) != 1 or len(facts.outputs) != 1:
        return None
    evidence.append(f"one input and one output pointer ({facts.name})")

    scalars = [p for p in facts.params if not p.is_pointer]
    if len(scalars) != 2:
        return None
    evidence.append(
        f"two scalar arguments ({', '.join(p.name for p in scalars)}), read as rows/cols"
    )
    return evidence


# Patterns a row-wise softmax can legitimately be written as. Serial is the
# naive form; tree and shuffle are the competent ones. Anything else is not a
# row-wise reduction at all and is refused before the evidence is examined.
_SOFTMAX_PATTERNS = (Pattern.REDUCE_SERIAL, Pattern.REDUCE_TREE, Pattern.REDUCE_SHUFFLE)


def propose(source: str, facts: KernelFacts, pattern: Pattern) -> Proposal | None:
    """The substitute to try for this kernel, or None to decline.

    Declining is a first-class outcome. `port` reports UNKNOWN and stops rather
    than reaching for the nearest kernel it happens to own.
    """
    if pattern in _SOFTMAX_PATTERNS:
        evidence = _looks_like_softmax(source, facts)
        if evidence:
            return Proposal(suite=ROW_SOFTMAX, evidence=evidence)
    return None


def infer_block(facts: KernelFacts) -> tuple[int, int, int]:
    """How many threads per block this kernel was written to be launched with.

    Launch geometry belongs to the kernel, not to the maths. Reusing the suite's
    geometry looked harmless and was not: row_softmax_tuned.cu launched with the
    naive kernel's block=(1,1,1) read uninitialised shared memory in its tree
    reduction, produced garbage, and the harness then reported the candidate as
    astronomically "better" than the wreckage. A wrong launch does not fail
    loudly, it manufactures a win, so it has to be got right.

    The evidence the parser already has:

      no threadIdx      one thread does the whole row, block=(1,1,1)
      a shared buffer   a tree reduction sized to the block, so block = its width
      otherwise         256, the ordinary choice, and --block overrides it
    """
    if not facts.uses_thread_index:
        return (1, 1, 1)
    for buf in facts.shared:
        if buf.size_bytes and "float" in buf.type:
            width = buf.size_bytes // 4
            if 1 <= width <= 1024:
                return (width, 1, 1)
    return (256, 1, 1)


def reference_launch(suite: Suite, facts: KernelFacts, block: tuple[int, int, int] | None = None):
    """The suite's launch spec, retargeted at the caller's kernel and geometry."""
    return replace(suite.launch, kernel=facts.name, block=block or infer_block(facts))


__all__ = ["Proposal", "propose", "reference_launch"]

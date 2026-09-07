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
from hipbridge.verify.suites import (
    LAYER_NORM,
    LAYER_NORM_AFFINE,
    RMS_NORM,
    ROW_SOFTMAX,
    Suite,
)


@dataclass(frozen=True)
class Proposal:
    """A substitute worth trying, and why it was proposed."""

    suite: Suite
    evidence: list[str]

    @property
    def name(self) -> str:
        return self.suite.name


def strip_comments(source: str) -> str:
    """Code only. Comments are prose and must not be treated as evidence.

    Found the hard way: rms_norm.cu explains in a comment that it "skips the
    mean entirely", and the LayerNorm test excluded it for containing the word
    mean, so the correct kernel was refused by its own documentation. The same
    hole runs the other way and matters more: a comment mentioning expf could
    talk this into proposing a softmax for a kernel that computes nothing of the
    kind. Evidence has to come from what the kernel does.
    """
    source = re.sub(r"/\*.*?\*/", " ", source, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", " ", source)


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

    return evidence


def _normalises_by_scale(source: str) -> bool:
    """Divides by a root of an accumulated quantity, however it is spelled."""
    return bool(re.search(r"\brsqrtf?\s*\(|\bsqrtf?\s*\(|\brsqrt\b", source))


def _looks_like_layer_norm(source: str, facts: KernelFacts) -> list[str] | None:
    """LayerNorm centres before it scales. That is what separates it from RMSNorm.

    Two accumulations and a subtraction of the first from every element, then a
    reciprocal square root. RMSNorm has the scale and not the centring, which is
    the whole difference between them and the one that must not be confused:
    substituting RMSNorm for LayerNorm changes the output of every non-zero-mean
    row, silently.
    """
    if not _normalises_by_scale(source):
        return None
    if not re.search(r"\bmean\b", source):
        return None
    # A deviation from the mean, spelled as a subtraction of it.
    if not re.search(r"-\s*mean\b", source):
        return None
    if facts.scalar_accumulations + facts.shared_accumulations < 2:
        return None
    return [
        "computes a mean and subtracts it, so it centres before scaling",
        "scales by a reciprocal square root of an accumulated quantity",
        f"{facts.scalar_accumulations + facts.shared_accumulations} accumulations, "
        "consistent with a mean pass and a variance pass",
    ]


def _looks_like_rms_norm(source: str, facts: KernelFacts) -> list[str] | None:
    """RMSNorm scales by the root mean square and never centres."""
    if not _normalises_by_scale(source):
        return None
    if re.search(r"-\s*mean\b|\bmean\b", source):
        return None  # centring means it is LayerNorm, not RMSNorm
    # A sum of squares: the same value multiplied by itself into an accumulator.
    if not re.search(r"(\w+)\s*\*\s*\1|\[i\]\s*\*\s*\w*\[i\]", source):
        return None
    return [
        "accumulates squares and never subtracts a mean",
        "scales by a reciprocal square root of that accumulation",
        "no centring pass, which is what separates RMSNorm from LayerNorm",
    ]


def operand_mismatch(suite: Suite, facts: KernelFacts) -> str | None:
    """Do the kernel's extra inputs line up with the substitute's, by name?

    Position is not enough. The generated driver binds input buffers in
    declaration order, so a kernel written `(in, beta, gamma, out, ...)` gets
    gamma where it expects beta. LayerNorm with its scale and shift exchanged
    still runs, still produces finite output, and is wrong on every row. The
    oracle would fail it and say only that the numbers disagreed.

    Returns None when the names line up, or a sentence naming the problem. A
    kernel using unfamiliar names is not refused for that alone: names are a
    weaker signal than the numeric proof, so an unrecognised spelling is
    allowed through while a recognised one in the wrong position is not.
    """
    if not suite.extras:
        return None

    got = [p.name for p in facts.inputs[1:]]
    if len(got) != len(suite.extras):
        return f"expected {len(suite.extras)} extra inputs, found {len(got)}"

    for i, (operand, name) in enumerate(zip(suite.extras, got, strict=True)):
        if operand.matches(name):
            continue
        # Does it belong to a different position? That is a swap, not a
        # vocabulary difference, and it is the dangerous case.
        for j, other in enumerate(suite.extras):
            if j != i and other.matches(name):
                return (
                    f"operand {i + 1} is named `{name}`, which this substitute "
                    f"expects at position {j + 1}: the weights appear to be in a "
                    f"different order than `{'`, `'.join(o.name for o in suite.extras)}`"
                )
    return None


# Patterns a row-wise reduction can legitimately be written as. Serial is the
# naive form; tree and shuffle are the competent ones. Anything else is not a
# row-wise reduction at all and is refused before the evidence is examined.
_ROW_PATTERNS = (Pattern.REDUCE_SERIAL, Pattern.REDUCE_TREE, Pattern.REDUCE_SHUFFLE)


def propose(
    source: str,
    facts: KernelFacts,
    pattern: Pattern,
    notes: list[str] | None = None,
) -> Proposal | None:
    """The substitute to try for this kernel, or None to decline.

    Declining is a first-class outcome. `port` reports UNKNOWN and stops rather
    than reaching for the nearest kernel it happens to own. Order matters only
    in that each test is exclusive of the others: softmax exponentiates,
    LayerNorm centres, RMSNorm does neither, and a kernel matching none of them
    gets no proposal at all.

    Pass `notes` to collect the reasons a near miss was declined. A kernel that
    matched every test but wired its weights in another order is the case worth
    explaining, because "no substitution proposed" would send someone hunting
    for a missing feature rather than reading their own signature.
    """
    if pattern not in _ROW_PATTERNS:
        return None

    source = strip_comments(source)
    scalars = [p for p in facts.params if not p.is_pointer]

    for suite, test in (
        (ROW_SOFTMAX, _looks_like_softmax),
        (LAYER_NORM, _looks_like_layer_norm),
        (LAYER_NORM_AFFINE, _looks_like_layer_norm),
        (RMS_NORM, _looks_like_rms_norm),
    ):
        evidence = test(source, facts)
        if not evidence:
            continue

        # The signature has to match the substitute's, which is how the plain
        # and affine forms of the same maths are told apart: they satisfy the
        # same evidence and differ only in taking weights. Substituting one for
        # the other would drop or invent a scale and shift, silently.
        wanted_inputs = 1 + len(suite.extras)
        if len(facts.inputs) != wanted_inputs or len(facts.outputs) != 1 or len(scalars) != 2:
            continue

        # Names, once the count is right. A recognised name in the wrong slot is
        # a swap, and a swap runs, returns finite numbers and is wrong.
        problem = operand_mismatch(suite, facts)
        if problem:
            if notes is not None:
                notes.append(f"{suite.name}: {problem}")
            continue

        if suite.extras:
            names = ", ".join(p.name for p in facts.inputs[1:])
            evidence = [*evidence, f"takes {len(suite.extras)} weight tensors ({names})"]
        return Proposal(suite=suite, evidence=evidence)
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


__all__ = ["Proposal", "infer_block", "propose", "reference_launch", "strip_comments"]

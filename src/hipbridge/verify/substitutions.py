"""Deciding whether a recognized kernel has a substitute worth proposing.

A pattern is not a licence to substitute. `row_softmax.cu` is recognized as
`reduce_serial`, and so is a kernel that sums a row, and so is one that takes a
product. Swapping a softmax into any of them would be a catastrophe that the
recognizer alone cannot prevent, because the pattern describes the *shape* of
the computation, not the maths.

So a proposal needs two things the recognizer does not supply:

  1. corroborating evidence, spelled out per substitution below, narrow enough
     that the guess is defensible
  2. a numeric proof, which is the part that actually makes it safe

Only the second one is load bearing. The evidence decides what to try; the
harness decides whether it was right, by compiling the caller's own kernel and
comparing both against a float64 oracle on device. A wrong guess fails there and
is reported as a failure, never as a substitution.

## Evidence comes from the parser, not from the source text

It used to be regular expressions over the .cu file, and that broke three times
in one week, every time on something that had nothing to do with the maths:

  - `rms_norm.cu` explained in a comment that it "skips the mean entirely", and
    the LayerNorm test excluded anything mentioning a mean, so the correct
    kernel was refused by its own documentation
  - making the kernels element-type generic turned `ri[i] * ri[i]` into
    `(float)ri[i] * (float)ri[i]`, and the sum-of-squares pattern stopped
    recognising RMSNorm
  - a test asserting the CI workflow has no schedule failed on the comment
    explaining why it has no schedule

The dangerous direction is the one that did not happen yet: a comment
mentioning expf talking this into proposing a softmax for a kernel that computes
nothing of the kind. Everything below now reads `KernelFacts`, so comments,
casts, whitespace and variable names cannot reach the decision at all.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from hipbridge.frontend.ir import KernelFacts, Pattern
from hipbridge.verify.suites import BUILTIN, Suite


@dataclass(frozen=True)
class Proposal:
    """A substitute worth trying, and why it was proposed."""

    suite: Suite
    evidence: list[str]
    # The epsilon both the substitute and the oracle get built with, read off
    # the caller's kernel. None for the suites whose maths has no epsilon.
    epsilon: float | None = None

    @property
    def name(self) -> str:
        return self.suite.name


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
        for j, other in enumerate(suite.extras):
            if j != i and other.matches(name):
                return (
                    f"operand {i + 1} is named `{name}`, which this substitute "
                    f"expects at position {j + 1}: the weights appear to be in a "
                    f"different order than `{'`, `'.join(o.name for o in suite.extras)}`"
                )
    return None


def propose(
    facts: KernelFacts,
    pattern: Pattern,
    notes: list[str] | None = None,
    epsilon: float | None = None,
) -> Proposal | None:
    """The substitute to try for this kernel, or None to decline.

    Declining is a first-class outcome. `port` reports UNKNOWN and stops rather
    than reaching for the nearest kernel it happens to own.

    Pass `epsilon` to declare the constant a normalisation adds when the kernel
    does not spell it as a literal the parser can read. It overrides whatever
    was parsed, because a caller who knows their own kernel is better evidence
    than an AST walk.

    Pass `notes` to collect the reasons a near miss was declined. A kernel that
    matched every test but wired its weights in another order is the case worth
    explaining, because "no substitution proposed" would send someone hunting
    for a missing feature rather than reading their own signature.
    """
    scalars = [p for p in facts.params if not p.is_pointer]

    # Each substitute declares the shapes its maths can legitimately take. RoPE
    # is the reason this is per-suite rather than one global gate: it is a map,
    # not a reduction, so the reduction patterns would exclude it and the
    # reduction suites must not claim a row_map.
    #
    # Iterated from BUILTIN rather than re-listed here. The old tuple was a
    # second registry of every suite, and a suite present in one and absent from
    # the other was unproposable with nothing to say so.
    for suite in BUILTIN:
        if suite.evidence is None or pattern not in suite.patterns:
            continue
        evidence = suite.evidence(facts)
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

        # An epsilon nobody could read is not a detail to default. Both the
        # substitute and the float64 oracle are built from it, so assuming this
        # project's constant rebuilds the truth around the wrong number: a
        # kernel correctly written with 1e-6 was scored the less accurate side
        # on every case, while the substitution moved its output by 68% on tiny
        # inputs and the report called that a 3.2e7x accuracy win.
        eps = epsilon if epsilon is not None else facts.epsilon
        if suite.uses_epsilon and eps is None:
            if notes is not None:
                notes.append(
                    f"{suite.name}: normalises by rsqrt(scale + eps), but no epsilon "
                    "literal could be read from the kernel, and substituting with an "
                    "assumed one changes every element. Declare it with --eps."
                )
            continue

        if suite.extras:
            names = ", ".join(p.name for p in facts.inputs[1:])
            evidence = [*evidence, f"takes {len(suite.extras)} weight tensors ({names})"]
        if suite.uses_epsilon:
            evidence = [
                *evidence,
                f"adds {eps:g} before the reciprocal square root, so the substitute "
                f"and the oracle are both built with {eps:g}",
            ]
        return Proposal(
            suite=suite,
            evidence=evidence,
            epsilon=eps if suite.uses_epsilon else None,
        )
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


__all__ = [
    "Proposal",
    "infer_block",
    "operand_mismatch",
    "propose",
    "reference_launch",
]

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
from hipbridge.verify.suites import (
    LAYER_NORM,
    LAYER_NORM_AFFINE,
    RMS_NORM,
    RMS_NORM_AFFINE,
    RMS_NORM_ROPE,
    ROPE,
    ROW_SOFTMAX,
    Suite,
)

# Spellings of the same operation across dialects and precisions.
_EXP = {"exp", "expf", "__expf", "exp2f", "hexp"}
_MAX = {"max", "fmax", "fmaxf", "fmaxf16", "hmax"}
_RSQRT = {"rsqrt", "rsqrtf", "sqrt", "sqrtf", "hrsqrt", "__frsqrt_rn"}


@dataclass(frozen=True)
class Proposal:
    """A substitute worth trying, and why it was proposed."""

    suite: Suite
    evidence: list[str]

    @property
    def name(self) -> str:
        return self.suite.name


def _calls_any(facts: KernelFacts, names: set[str]) -> bool:
    return any(c in names for c in facts.calls)


def _reduced_quantities(facts: KernelFacts) -> int:
    """How many separate quantities the kernel reduces over the row.

    LayerNorm needs two, a mean and a variance. RMSNorm needs one. That holds
    however either is written: naively they show up as scalar accumulations, and
    in a tuned version as the shared buffers a block reduction needs. Counting
    them is what separates the two without asking whether a variable happens to
    be called `mean`, and a kernel using `mu` is judged the same as one that
    does not.
    """
    return max(facts.scalar_accumulations, len(facts.shared))


def _looks_like_softmax(facts: KernelFacts) -> list[str] | None:
    """Exponentiation plus a maximum. No other row-wise reduction has both.

    A row sum has neither, a LayerNorm has neither, and a logsumexp has both and
    would be proposed here, tried, and rejected by the oracle. That is the
    intended division of labour.
    """
    if not _calls_any(facts, _EXP):
        return None
    if not _calls_any(facts, _MAX):
        return None
    return [
        f"calls {', '.join(c for c in facts.calls if c in _EXP)}, so it is not a plain sum",
        f"calls {', '.join(c for c in facts.calls if c in _MAX)}, the usual stability pass",
    ]


def _looks_like_layer_norm(facts: KernelFacts) -> list[str] | None:
    """Scales by a reciprocal square root, and reduces two quantities."""
    if _calls_any(facts, _EXP):
        return None  # exponentiating makes it a softmax, not a normalisation
    if not _calls_any(facts, _RSQRT):
        return None
    n = _reduced_quantities(facts)
    if n < 2:
        return None
    return [
        "scales by a reciprocal square root of an accumulated quantity",
        f"reduces {n} quantities over the row, consistent with a mean and a variance",
        "centres before scaling, which is what separates it from RMSNorm",
    ]


def _looks_like_rms_norm(facts: KernelFacts) -> list[str] | None:
    """Scales by a root mean square, reducing one quantity and never centring.

    The confusion that must not happen: substituting RMSNorm for LayerNorm
    changes the output of every row whose mean is not zero, silently. One
    reduced quantity against two is the structural difference between them.
    """
    if _calls_any(facts, _EXP):
        return None
    if not _calls_any(facts, _RSQRT):
        return None
    if _reduced_quantities(facts) != 1:
        return None
    return [
        "scales by a reciprocal square root of an accumulated quantity",
        "reduces exactly one quantity, so it never centres",
        "no mean pass, which is what separates RMSNorm from LayerNorm",
    ]


def _looks_like_rope(facts: KernelFacts) -> list[str] | None:
    """A per-position map that accumulates nothing and scales by nothing.

    RoPE cannot be identified by what it accumulates, because it accumulates
    nothing, and the pattern gate has already established that. What remains is
    the absence of the operations that would make it something else: no
    exponential, no reciprocal square root, no reduction. Combined with the
    signature check it is narrow enough to try, and the oracle settles it.
    """
    if _calls_any(facts, _EXP) or _calls_any(facts, _RSQRT):
        return None
    if facts.scalar_accumulations or facts.shared_accumulations:
        return None
    return [
        "accumulates nothing across the row, so it is a map and not a reduction",
        "calls neither an exponential nor a reciprocal square root",
        "takes per-position tables alongside the tensor it transforms",
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

# Kernels that walk a row without reducing it. RoPE lives here.
_MAP_PATTERNS = (Pattern.ROW_MAP,)


def propose(
    facts: KernelFacts,
    pattern: Pattern,
    notes: list[str] | None = None,
) -> Proposal | None:
    """The substitute to try for this kernel, or None to decline.

    Declining is a first-class outcome. `port` reports UNKNOWN and stops rather
    than reaching for the nearest kernel it happens to own.

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
    for suite, patterns, test in (
        (ROW_SOFTMAX, _ROW_PATTERNS, _looks_like_softmax),
        (LAYER_NORM, _ROW_PATTERNS, _looks_like_layer_norm),
        (LAYER_NORM_AFFINE, _ROW_PATTERNS, _looks_like_layer_norm),
        (RMS_NORM, _ROW_PATTERNS, _looks_like_rms_norm),
        (RMS_NORM_AFFINE, _ROW_PATTERNS, _looks_like_rms_norm),
        # Same evidence as RMSNorm, and the signature is what separates them: a
        # scale plus two position tables is a normalisation fused with a
        # rotation, and nothing else in this set takes three operands. The
        # rotation itself leaves no structural trace, so the arity carries the
        # discrimination and the oracle carries the proof.
        (RMS_NORM_ROPE, _ROW_PATTERNS, _looks_like_rms_norm),
        (ROPE, _MAP_PATTERNS, _looks_like_rope),
    ):
        if pattern not in patterns:
            continue
        evidence = test(facts)
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


__all__ = [
    "Proposal",
    "infer_block",
    "operand_mismatch",
    "propose",
    "reference_launch",
]

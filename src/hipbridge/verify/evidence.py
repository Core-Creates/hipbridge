"""Corroborating evidence: does this kernel's maths match a substitute's?

Split out of `substitutions.py` so a `Suite` can carry its own evidence test as
a field. That inverts the old arrangement, where `propose()` held a tuple
re-listing every suite alongside its test, and a suite added to `BUILTIN` was
silently unproposable until someone remembered to add a row there too.

A pattern is not a licence to substitute: `row_softmax.cu` is `reduce_serial`,
and so is a kernel that sums a row. The tests below are what narrow a pattern to
a specific piece of arithmetic, and they are deliberately not the thing that
makes a substitution safe. The float64 oracle is. Evidence decides what to try.

Reads `KernelFacts` and nothing else, so it imports no torch and stays usable
wherever the frontend is. Everything here used to be regular expressions over
the .cu text, which broke three times in one week on comments, casts and
whitespace: `rms_norm.cu` was refused by its own comment explaining that it
"skips the mean entirely".
"""

from __future__ import annotations

from hipbridge.frontend.ir import KernelFacts, Pattern
from hipbridge.frontend.prelude import EXP_NAMES, MAX_NAMES, NORMALISING

# Patterns a row-wise reduction can legitimately be written as. Serial is the
# naive form; tree and shuffle are the competent ones. Anything else is not a
# row-wise reduction at all and is refused before the evidence is examined.
ROW_PATTERNS = (Pattern.REDUCE_SERIAL, Pattern.REDUCE_TREE, Pattern.REDUCE_SHUFFLE)

# Kernels that walk a row without reducing it. RoPE lives here.
MAP_PATTERNS = (Pattern.ROW_MAP,)


def calls_any(facts: KernelFacts, names: frozenset[str]) -> bool:
    return any(c in names for c in facts.calls)


def reduced_quantities(facts: KernelFacts) -> int:
    """How many separate quantities the kernel reduces over the row.

    LayerNorm needs two, a mean and a variance. RMSNorm needs one. That holds
    however either is written: naively they show up as scalar accumulations, and
    in a tuned version as the shared buffers a block reduction needs. Counting
    them is what separates the two without asking whether a variable happens to
    be called `mean`, and a kernel using `mu` is judged the same as one that
    does not.
    """
    return max(facts.scalar_accumulations, len(facts.shared))


def looks_like_softmax(facts: KernelFacts) -> list[str] | None:
    """Exponentiation plus a maximum. No other row-wise reduction has both.

    A row sum has neither, a LayerNorm has neither, and a logsumexp has both and
    would be proposed here, tried, and rejected by the oracle. That is the
    intended division of labour.
    """
    if not calls_any(facts, EXP_NAMES):
        return None
    if not calls_any(facts, MAX_NAMES):
        return None
    return [
        f"calls {', '.join(c for c in facts.calls if c in EXP_NAMES)}, so it is not a plain sum",
        f"calls {', '.join(c for c in facts.calls if c in MAX_NAMES)}, the usual stability pass",
    ]


def looks_like_layer_norm(facts: KernelFacts) -> list[str] | None:
    """Scales by a reciprocal square root, and reduces two quantities."""
    if calls_any(facts, EXP_NAMES):
        return None  # exponentiating makes it a softmax, not a normalisation
    if not calls_any(facts, NORMALISING):
        return None
    n = reduced_quantities(facts)
    if n < 2:
        return None
    return [
        "scales by a reciprocal square root of an accumulated quantity",
        f"reduces {n} quantities over the row, consistent with a mean and a variance",
        "centres before scaling, which is what separates it from RMSNorm",
    ]


def looks_like_rms_norm(facts: KernelFacts) -> list[str] | None:
    """Scales by a root mean square, reducing one quantity and never centring.

    The confusion that must not happen: substituting RMSNorm for LayerNorm
    changes the output of every row whose mean is not zero, silently. One
    reduced quantity against two is the structural difference between them.
    """
    if calls_any(facts, EXP_NAMES):
        return None
    if not calls_any(facts, NORMALISING):
        return None
    if reduced_quantities(facts) != 1:
        return None
    return [
        "scales by a reciprocal square root of an accumulated quantity",
        "reduces exactly one quantity, so it never centres",
        "no mean pass, which is what separates RMSNorm from LayerNorm",
    ]


def looks_like_rope(facts: KernelFacts) -> list[str] | None:
    """A per-position map that accumulates nothing and scales by nothing.

    RoPE cannot be identified by what it accumulates, because it accumulates
    nothing, and the pattern gate has already established that. What remains is
    the absence of the operations that would make it something else: no
    exponential, no reciprocal square root, no reduction. Combined with the
    signature check it is narrow enough to try, and the oracle settles it.
    """
    if calls_any(facts, EXP_NAMES) or calls_any(facts, NORMALISING):
        return None
    if facts.scalar_accumulations or facts.shared_accumulations:
        return None
    return [
        "accumulates nothing across the row, so it is a map and not a reduction",
        "calls neither an exponential nor a reciprocal square root",
        "takes per-position tables alongside the tensor it transforms",
    ]


__all__ = [
    "MAP_PATTERNS",
    "ROW_PATTERNS",
    "calls_any",
    "looks_like_layer_norm",
    "looks_like_rms_norm",
    "looks_like_rope",
    "looks_like_softmax",
    "reduced_quantities",
]

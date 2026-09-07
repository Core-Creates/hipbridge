"""Recognizer registry.

A recognizer inspects KernelFacts and either claims the kernel or declines.
There is no catch-all rule and no default pattern: if every recognizer declines,
the answer is UNKNOWN. That is the whole point of this module.
"""

from __future__ import annotations

from collections.abc import Callable

from hipbridge.frontend.ir import KernelFacts, Pattern, Recognition

# A rule returns (pattern, confidence, rationale) or None to decline.
#
# Confidence is reported, never acted on. See Recognition.confidence: gating a
# substitution on it would reject correct kernels on weaker evidence than the
# numeric proof that follows, so it exists to inform a reader and nothing else.
Rule = Callable[[KernelFacts], tuple[Pattern, str, list[str]] | None]

_RULES: list[tuple[int, str, Rule]] = []


def rule(name: str, priority: int = 100) -> Callable[[Rule], Rule]:
    """Register a recognizer. Lower priority runs first."""

    def wrap(fn: Rule) -> Rule:
        _RULES.append((priority, name, fn))
        _RULES.sort(key=lambda t: (t[0], t[1]))
        return fn

    return wrap


def registered() -> list[str]:
    return [name for _, name, _ in _RULES]


def recognize(facts: KernelFacts) -> Recognition:
    """Run every registered rule in priority order. First claim wins."""
    for _, name, fn in _RULES:
        verdict = fn(facts)
        if verdict is None:
            continue
        pattern, confidence, rationale = verdict
        return Recognition(
            facts=facts,
            pattern=pattern,
            confidence=confidence,
            rationale=[f"[{name}] {r}" for r in rationale],
        )

    return Recognition(
        facts=facts,
        pattern=Pattern.UNKNOWN,
        confidence="none",
        rationale=[
            f"no recognizer claimed this kernel ({len(_RULES)} tried)",
            f"observed: {facts.loops} loop(s), {len(facts.shared)} shared buffer(s), "
            f"{facts.barriers} barrier(s), {len(facts.shuffle_intrinsics)} shuffle(s)",
        ],
    )

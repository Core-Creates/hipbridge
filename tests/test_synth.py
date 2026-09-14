"""The generate-and-verify loop.

Everything here runs on CPU with a fake generator and a fake gate. The point of
the loop is that the gate decides, so these tests are about what happens to
candidates, not about whether any particular model writes good kernels.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from hipbridge import synth

GOOD = "def candidate(x):\n    return x * 2\n"
NO_ENTRY_POINT = "def something_else(x):\n    return x\n"
SYNTAX_ERROR = "def candidate(x:\n    return x\n"
RAISES = "def candidate(x):\n    raise ValueError('kernel exploded')\n"


@dataclass
class FakeSummary:
    ok: bool
    failures: list = None  # noqa: RUF012 - mirrors the harness shape loosely


def _gate(passing: set[str]):
    """A stand-in for the harness: passes candidates whose output matches."""

    def verify_one(fn):
        return FakeSummary(ok=fn(2) in passing)

    return verify_one


def test_generated_code_does_not_run_without_consent():
    """Executing a model's output is a decision, and it has to be made explicitly.

    Not a theoretical concern: a generated kernel runs with the privileges of the
    process, on machines that in this project's case hold cloud credentials.
    """
    with pytest.raises(synth.UntrustedCodeError):
        synth.load(synth.Candidate(GOOD))

    fn = synth.load(synth.Candidate(GOOD), allow_untrusted_code=True)
    assert fn(21) == 42


def test_the_search_refuses_rather_than_skipping_the_gate():
    """The refusal must stop the loop, not be recorded as one more failed attempt."""
    with pytest.raises(synth.UntrustedCodeError):
        synth.search(synth.StaticGenerator([GOOD]), "prompt", _gate({4}), count=1)


def test_a_candidate_that_passes_is_kept():
    found = synth.search(
        synth.StaticGenerator([GOOD]),
        "prompt",
        _gate({4}),
        count=1,
        allow_untrusted_code=True,
    )
    assert found.winner is not None
    assert found.winner.status == "passed"
    assert "kept" in str(found)


def test_every_way_a_candidate_can_fail_is_recorded_not_raised():
    """Broken proposals are data. A generator that fails the same way twenty
    times is saying something about the prompt, and that is only visible if the
    failures are kept."""
    found = synth.search(
        synth.StaticGenerator([SYNTAX_ERROR, NO_ENTRY_POINT, RAISES, GOOD]),
        "prompt",
        _gate({4}),
        count=4,
        allow_untrusted_code=True,
    )

    statuses = [a.status for a in found.attempts]
    assert statuses == ["unloadable", "unloadable", "raised", "passed"]
    assert found.winner is not None
    assert any("SyntaxError" in a.detail for a in found.attempts)
    assert any("candidate" in a.detail for a in found.attempts), "missing entry point named"


def test_a_wrong_kernel_is_rejected_even_though_it_runs():
    """The interesting failure: it loads, it runs, it returns numbers, it is wrong.

    This is the only case the gate is really for. The other three are caught by
    Python before any arithmetic happens.
    """
    wrong = "def candidate(x):\n    return x * 3\n"
    found = synth.search(
        synth.StaticGenerator([wrong]),
        "prompt",
        _gate({4}),
        count=1,
        allow_untrusted_code=True,
    )

    assert found.winner is None
    assert found.attempts[0].status == "rejected"
    assert "nothing passed" in str(found)


def test_the_search_stops_at_the_first_survivor():
    """Later candidates are not run once one has been proved."""
    found = synth.search(
        synth.StaticGenerator([GOOD, GOOD, GOOD]),
        "prompt",
        _gate({4}),
        count=3,
        allow_untrusted_code=True,
    )
    assert len(found.attempts) == 1


def test_markdown_fences_are_stripped():
    """Models wrap code in fences, and a fence is a syntax error."""
    fenced = "Here you go:\n```python\ndef candidate(x):\n    return x\n```\nHope that helps!"
    body = synth._strip_fences(fenced)

    assert body.strip().startswith("def candidate")
    assert "```" not in body
    assert "Hope that helps" not in body


def test_the_prompt_names_the_contract():
    """A generator that returns a differently shaped thing fails at the loader
    for reasons unrelated to the maths, so the prompt has to be specific."""
    prompt = synth.prompt_for("__global__ void k() {}", "row_softmax", "gfx942")

    assert synth.ENTRY_POINT in prompt
    assert "row_softmax" in prompt
    assert "gfx942" in prompt
    assert "64" in prompt, "wavefront width has to be stated"
    assert "float64 oracle" in prompt, "the generator should know how it will be judged"


def test_the_prompt_states_the_width_of_the_arch_it_names():
    """It told every generator 64 wide, including one asked to target RDNA."""
    prompt = synth.prompt_for("__global__ void k() {}", "row_softmax", "gfx1100")

    assert "gfx1100" in prompt
    assert "Wavefronts are 32 wide" in prompt
    assert "64 wide" not in prompt
    assert "CDNA" not in prompt


@pytest.mark.parametrize("arch", ["", "gfx906"])
def test_the_prompt_does_not_invent_a_width_it_does_not_know(arch):
    prompt = synth.prompt_for("__global__ void k() {}", "row_softmax", arch)

    assert "not known" in prompt
    assert "Wavefronts are" not in prompt
    assert "gfx942" not in prompt, "an unnamed arch is not the MI300X"


def test_identical_proposals_are_recognisable():
    """Sampling the same model repeatedly returns duplicates; they should be
    visible as duplicates rather than counted as independent attempts."""
    a = synth.Candidate(GOOD, origin="m", index=0)
    b = synth.Candidate(GOOD, origin="m", index=1)

    assert a.digest == b.digest
    assert a.describe() != b.describe()


def test_a_generator_that_fails_ends_the_search_quietly():
    """An unavailable model should stop the loop, not take the process with it."""
    gen = synth.ScriptGenerator(command=["definitely-not-a-real-command-xyz"])
    found = synth.search(gen, "prompt", _gate({4}), count=3, allow_untrusted_code=True)

    assert found.attempts == []
    assert found.winner is None

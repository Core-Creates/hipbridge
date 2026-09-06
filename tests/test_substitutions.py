"""What `port` is willing to propose, and what it refuses to.

A recognized pattern is not a licence to substitute. `reduce_serial` covers a
row sum, a row product and a softmax alike, and swapping a softmax into the
first two would be a disaster no recognizer can prevent. These tests pin the
refusals, because the refusals are the safety property.
"""

from __future__ import annotations

import pytest

from hipbridge import parse_file, recognize
from hipbridge.verify import available

pytestmark = pytest.mark.skipif(not available(), reason="[verify] extra not installed")


def _propose(examples, name):
    from hipbridge.verify import substitutions

    kernels = parse_file(examples / name)
    facts = kernels[0]
    result = recognize(facts)
    source = (examples / name).read_text(encoding="utf-8")
    return substitutions.propose(source, facts, result.pattern), facts


def test_a_naive_softmax_is_proposed(examples):
    proposal, _ = _propose(examples, "row_softmax.cu")
    assert proposal is not None
    assert proposal.name == "row_softmax"
    assert any("expf" in e for e in proposal.evidence)


def test_a_competent_softmax_is_also_proposed(examples):
    """The same maths written well must be recognized as the same maths."""
    proposal, _ = _propose(examples, "row_softmax_tuned.cu")
    assert proposal is not None
    assert proposal.name == "row_softmax"


@pytest.mark.parametrize("name", ["tree_reduce.cu", "saxpy.cu", "tiled_transpose.cu"])
def test_everything_else_is_refused(examples, name):
    """Sharing a pattern with softmax is not evidence of being softmax.

    tree_reduce.cu is a reduce_tree exactly as row_softmax_tuned.cu is, and it
    sums rather than normalising. Proposing a softmax for it would corrupt data
    silently, so the evidence test has to be narrower than the pattern.
    """
    proposal, _ = _propose(examples, name)
    assert proposal is None, f"{name} must not attract a substitution"


def test_launch_geometry_comes_from_the_kernel_not_the_suite(examples):
    """The bug this test exists for produced a fabricated proof.

    row_softmax_tuned.cu launched with the naive kernel's block=(1,1,1) read
    uninitialised shared memory in its tree reduction and returned garbage. The
    harness then scored the candidate 3.9e75x "better" than that garbage and
    reported a substitution as proved. Geometry belongs to the kernel.
    """
    from hipbridge.verify import substitutions

    _, naive = _propose(examples, "row_softmax.cu")
    _, tuned = _propose(examples, "row_softmax_tuned.cu")

    assert substitutions.infer_block(naive) == (1, 1, 1), "one thread per row, as written"
    assert substitutions.infer_block(tuned) == (256, 1, 1), "sized by its shared buffer"

    from hipbridge.verify.suites import ROW_SOFTMAX

    launch = substitutions.reference_launch(ROW_SOFTMAX, tuned)
    assert launch.kernel == "row_softmax_tuned", "must target the caller's kernel"
    assert launch.block == (256, 1, 1)

    override = substitutions.reference_launch(ROW_SOFTMAX, tuned, (64, 1, 1))
    assert override.block == (64, 1, 1), "--block has to win over inference"

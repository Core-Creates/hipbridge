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

    facts = parse_file(examples / name)[0]
    return substitutions.propose(facts, recognize(facts).pattern), facts


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


def test_layer_norm_and_rms_norm_are_not_confused(examples):
    """The one confusion that would corrupt data silently.

    RMSNorm scales without centring. Substituting it for LayerNorm changes the
    output of every row whose mean is not zero, and nothing about the shape of
    the two kernels distinguishes them: both are row-wise reductions ending in a
    reciprocal square root.
    """
    ln, _ = _propose(examples, "layer_norm.cu")
    rn, _ = _propose(examples, "rms_norm.cu")

    assert ln is not None and ln.name == "layer_norm"
    assert rn is not None and rn.name == "rms_norm"
    assert any("separates it from RMSNorm" in e for e in ln.evidence)
    assert any("never centres" in e for e in rn.evidence)
    assert any("reduces 2 quantities" in e for e in ln.evidence), (
        "the count is the structural difference, so it belongs in the evidence"
    )


def test_the_tuned_norms_map_to_the_same_substitutes(examples):
    """Written well or written naively, it is still the same maths."""
    for name, expected in (
        ("layer_norm_tuned.cu", "layer_norm"),
        ("rms_norm_tuned.cu", "rms_norm"),
    ):
        proposal, _ = _propose(examples, name)
        assert proposal is not None, f"{name} should be recognized as {expected}"
        assert proposal.name == expected


def test_comments_and_formatting_cannot_reach_the_decision():
    """Evidence is now read off the parser, so prose is structurally excluded.

    This test used to check that comments were stripped before matching. The
    stripping worked and the approach did not: three separate regressions came
    from text that had nothing to do with the maths, including a kernel refused
    because its own comment said "skips the mean entirely" and a sum-of-squares
    pattern broken by an added cast.

    The dangerous direction is the one that never happened: a comment mentioning
    expf talking the tool into proposing a softmax. It cannot now, because
    comments are not part of a KernelFacts.
    """
    from hipbridge.frontend.ir import KernelFacts, Param, Pattern
    from hipbridge.verify import substitutions

    def facts(**kw):
        return KernelFacts(
            name="k",
            params=[
                Param("in", "const float *", True, True),
                Param("out", "float *", True, False),
                Param("rows", "int", False, False),
                Param("cols", "int", False, False),
            ],
            **kw,
        )

    # Says nothing, calls nothing: no proposal, whatever it claims in prose.
    assert substitutions.propose(facts(), Pattern.REDUCE_SERIAL) is None

    # Calls expf and a maximum: softmax, whatever it is named or commented.
    proposal = substitutions.propose(
        facts(calls=["expf", "fmaxf"], scalar_accumulations=1), Pattern.REDUCE_SERIAL
    )
    assert proposal is not None
    assert proposal.name == "row_softmax"


def test_the_norms_are_separated_by_how_many_quantities_they_reduce():
    """LayerNorm needs a mean and a variance. RMSNorm needs one sum.

    That holds however either is written, naively as scalar accumulations or in
    a tuned version as the shared buffers a block reduction needs, so it does
    not depend on a variable being called `mean`. A kernel using `mu` is judged
    the same as one that does not.
    """
    from hipbridge.frontend.ir import KernelFacts, Param, Pattern
    from hipbridge.verify import substitutions

    def facts(**kw):
        return KernelFacts(
            name="k",
            params=[
                Param("in", "const float *", True, True),
                Param("out", "float *", True, False),
                Param("rows", "int", False, False),
                Param("cols", "int", False, False),
            ],
            calls=["rsqrtf"],
            **kw,
        )

    two = substitutions.propose(facts(scalar_accumulations=2), Pattern.REDUCE_SERIAL)
    one = substitutions.propose(facts(scalar_accumulations=1), Pattern.REDUCE_SERIAL)

    assert two is not None and two.name == "layer_norm"
    assert one is not None and one.name == "rms_norm"

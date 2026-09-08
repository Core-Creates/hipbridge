"""The epsilon a normalisation adds has to come from the caller's kernel.

1e-5 and 1e-6 produce the same AST, so no amount of structural evidence can
tell them apart. Before this, the oracle and the substitute were both hardcoded
to 1e-5 and the caller's kernel was the only side that used its own value: a
correct kernel written with 1e-6 was judged `better` by the substitute on every
case, and on the TINY distribution the substitution moved the output by 68%
while the arbitration reported it as a 3.2e7x accuracy win.
"""

from __future__ import annotations

import pathlib

import pytest

from hipbridge import verify
from hipbridge.frontend.parser import parse_file, parse_source
from hipbridge.recognize import recognize

needs_verify = pytest.mark.skipif(not verify.available(), reason="[verify] extra not installed")

EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / "examples"

# Spelled here rather than imported: this half of the file must run on a core
# install, and hipbridge.verify.suites imports torch. The tie back to
# DEFAULT_EPS is asserted under the extra, below.
EXPECTED_DEFAULT_EPS = 1e-5

RMS_NORM = """
__global__ void rms_norm(const float *in, float *out, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float *ri = in + (size_t)row * cols;
    float *ro = out + (size_t)row * cols;
    %s
    float acc = 0.0f;
    for (int i = 0; i < cols; i++) acc += ri[i] * ri[i];
    float inv = %s;
    for (int i = 0; i < cols; i++) ro[i] = ri[i] * inv;
}
"""


def _rms_norm(decl: str = "", scale: str = "rsqrtf(acc / (float)cols + 1e-5f)"):
    return parse_source(RMS_NORM % (decl, scale))[0]


# --- torch-free: the parser half runs on a core install --------------------


def test_epsilon_is_read_off_the_kernel():
    assert _rms_norm(scale="rsqrtf(acc / (float)cols + 1e-6f)").epsilon == pytest.approx(1e-6)
    assert _rms_norm(scale="rsqrtf(acc / (float)cols + 1e-5f)").epsilon == pytest.approx(1e-5)
    assert _rms_norm(scale="1.0f / sqrtf(acc / (float)cols + 1e-12f)").epsilon == pytest.approx(
        1e-12
    )


def test_an_epsilon_that_is_not_a_literal_reads_as_absent():
    """Absent, not assumed. A guessed epsilon is a silent change of maths.

    The control matters: the two sources differ only in where the literal sits,
    so a None here cannot be a parse that fell over. Asserting `parse_errors ==
    0` instead was environment-dependent, passing on one libclang and reporting
    2 on CI's.
    """
    hidden = _rms_norm(decl="float e = 1e-6f;", scale="rsqrtf(acc / (float)cols + e)")
    inline = _rms_norm(scale="rsqrtf(acc / (float)cols + 1e-6f)")

    assert hidden.epsilon is None
    assert inline.epsilon == pytest.approx(1e-6)
    assert hidden.name == inline.name == "rms_norm"
    assert hidden.scalar_accumulations == inline.scalar_accumulations


def test_two_constants_in_one_call_cannot_be_told_apart():
    """Which addend is the epsilon is not decidable, so neither is claimed."""
    assert _rms_norm(scale="rsqrtf(acc * 0.5f + 1e-5f)").epsilon is None


def test_a_kernel_with_no_normalisation_has_no_epsilon():
    for name in ("row_softmax.cu", "rope.cu", "saxpy.cu"):
        for facts in parse_file(EXAMPLES / name):
            assert facts.epsilon is None, name


def test_every_shipped_example_normalises_with_the_default():
    """Guards the built-in suites, which run against these sources with DEFAULT_EPS.

    `verify` does not parse the example it measures, so a stray edit to one of
    these literals would put the oracle and the kernel on different constants
    with nothing to say so. This is that alarm, and it costs no GPU time.
    """
    normalising = sorted(p for p in EXAMPLES.glob("*norm*.cu"))
    assert normalising, "no normalising examples found"
    for path in normalising:
        for facts in parse_file(path):
            assert facts.epsilon == pytest.approx(EXPECTED_DEFAULT_EPS), path.name


# --- needs the extra -------------------------------------------------------


@needs_verify
def test_a_kernel_without_a_readable_epsilon_is_not_substituted():
    from hipbridge.verify import substitutions

    facts = _rms_norm(decl="float e = 1e-6f;", scale="rsqrtf(acc / (float)cols + e)")
    notes: list[str] = []
    assert substitutions.propose(facts, recognize(facts).pattern, notes) is None
    assert any("epsilon" in n for n in notes), notes


@needs_verify
def test_a_declared_epsilon_overrides_the_parse():
    """--eps exists for the kernel whose constant hides behind a macro or a variable."""
    from hipbridge.verify import substitutions

    facts = _rms_norm(decl="float e = 1e-6f;", scale="rsqrtf(acc / (float)cols + e)")
    proposal = substitutions.propose(facts, recognize(facts).pattern, None, epsilon=1e-6)
    assert proposal is not None
    assert proposal.epsilon == pytest.approx(1e-6)


@needs_verify
def test_suites_without_an_epsilon_do_not_ask_for_one():
    from hipbridge.verify import suites

    by_name = {s.name: s.uses_epsilon for s in suites.BUILTIN}
    assert by_name["row_softmax"] is False
    assert by_name["rope"] is False
    assert all(by_name[n] for n in ("layer_norm", "rms_norm", "rms_norm_affine", "rms_norm_rope"))


@needs_verify
def test_the_oracle_follows_the_epsilon_it_is_given():
    """And the difference is a scale factor, not rounding.

    Measured on values small enough that the mean square underflows the epsilon,
    which is where the whole output reduces to `x * rsqrt(eps)`. The two oracles
    then differ by sqrt(1e-5 / 1e-6), a factor of 3.16. On N(0,1) the same change
    moves the fourth decimal and slips under torch.allclose, which is how a
    constant nobody threaded through went unnoticed.
    """
    import math

    import torch

    from hipbridge.verify import suites

    x = torch.full((1, 4), 1e-30)
    loose = suites._rms_norm_oracle(x, eps=1e-5)
    tight = suites._rms_norm_oracle(x, eps=1e-6)
    assert float((tight / loose).max()) == pytest.approx(math.sqrt(10.0), rel=1e-6)


@needs_verify
def test_a_correct_kernel_is_no_longer_scored_worse_than_the_substitute():
    """The regression this whole change exists for.

    The caller's kernel and the substitute now compute the same maths, so the
    arbitration has nothing to separate them. Before, the substitute was
    `better` on every distribution because the oracle had been built around its
    constant rather than around the caller's.
    """
    import torch

    from hipbridge.verify import compare, inputs, substitutions, suites

    facts = _rms_norm(scale="rsqrtf(acc / (float)cols + 1e-6f)")
    proposal = substitutions.propose(facts, recognize(facts).pattern)
    assert proposal is not None and proposal.epsilon == pytest.approx(1e-6)

    oracle = suites.oracle_for(proposal.suite, proposal.epsilon)
    candidate, described = suites.candidate_for(proposal.suite, proposal.epsilon)
    assert "eps=1e-06" in described

    for distribution in (inputs.Distribution.TINY, inputs.Distribution.NORMAL):
        spec = inputs.InputSpec(shape=(4, 256), distribution=distribution)
        x = inputs.generate(spec)
        theirs = x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + 1e-6)
        arb = compare.arbitrate(candidate(x), theirs, oracle(x.double()))
        assert arb.verdict == "equivalent", (distribution, arb)


@needs_verify
def test_every_epsilon_suite_can_actually_be_bound():
    """The test that was missing when a metered run found the gap instead.

    `oracle_for` binds eps with functools.partial, so an oracle that does not
    take the argument raises TypeError at call time rather than at bind time.
    Only `port` calls it, `_cmd_port` has no tests, and `verify` uses the
    unbound oracle, so four of the five suites were exercised and the fifth was
    not: `_layer_norm_affine_oracle` reached an MI300X without an eps parameter
    and failed there with a raw traceback, 17 minutes into a paid run.

    This calls every eps-carrying suite the way port does, on CPU, for free.
    """
    import torch

    from hipbridge.verify import suites

    bound = [s for s in suites.BUILTIN if s.uses_epsilon]
    assert len(bound) == 5, [s.name for s in bound]

    for suite in bound:
        ins = suites.make_inputs(suite, suites.InputSpec((4, 8)))
        doubles = tuple(t.double() for t in ins)

        loose = suites.oracle_for(suite, 1e-5)(*doubles)
        tight = suites.oracle_for(suite, 1e-9)(*doubles)
        assert loose.shape == ins[0].shape, suite.name
        # eps has to reach the arithmetic, not merely be accepted by the call.
        assert not torch.equal(loose, tight), f"{suite.name}: oracle ignored eps"

        candidate, described = suites.candidate_for(suite, 1e-6)
        assert "eps=1e-06" in described, described
        assert candidate(*ins).shape == ins[0].shape, suite.name


@needs_verify
def test_the_default_matches_what_the_examples_spell():
    """Ties the torch-free example check above to the constant the suites use."""
    from hipbridge.verify.suites import DEFAULT_EPS

    assert DEFAULT_EPS == pytest.approx(EXPECTED_DEFAULT_EPS)

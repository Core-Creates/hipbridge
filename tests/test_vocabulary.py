"""Every spelling a recognizer tests for must be one the frontend can resolve.

`facts.calls` holds the names clang resolved, and nothing else. A spelling the
prelude does not declare cannot appear there, so a rule keying on it can never
fire. That is not a rule that fails loudly: the call leaves no CALL_EXPR at all,
the kernel simply looks smaller than it is, and the user is told "no recognizer
claimed this kernel".

The vocabulary used to live in two places, verbatim, in two layers:
`frontend/parser.py` held the normalising set and `verify/substitutions.py` held
its own copy plus exp and max. Neither was ever reconciled against the prelude,
and 13 of the 17 spellings between them were undeclarable. A softmax written
with `__expf`, which is what production CUDA writes, produced `calls=[]`.

So the prelude owns the sets and generates its declarations from them, and these
tests pin the containment in both directions.
"""

from __future__ import annotations

import pytest

from hipbridge import parse_source, recognize
from hipbridge.frontend import prelude

VOCABULARY = {
    "EXP_NAMES": prelude.EXP_NAMES,
    "MAX_NAMES": prelude.MAX_NAMES,
    "NORMALISING": prelude.NORMALISING,
    "SHUFFLE_NAMES": prelude.SHUFFLE_NAMES,
    "ATOMIC_NAMES": prelude.ATOMIC_NAMES,
}


@pytest.mark.parametrize("name", sorted(VOCABULARY))
def test_every_vocabulary_spelling_is_declared(name: str):
    """The containment that makes the vocabulary real rather than aspirational."""
    missing = VOCABULARY[name] - prelude.declared_names()
    assert not missing, (
        f"{name} contains {sorted(missing)}, which the prelude does not declare. "
        f"A spelling clang cannot resolve never reaches facts.calls, so any rule "
        f"keying on it is dead code."
    )


def test_the_frontend_and_the_recognizer_share_one_definition():
    """Not two equal copies. One object, so they cannot drift apart."""
    from hipbridge.frontend import parser
    from hipbridge.verify import evidence

    assert parser.NORMALISING is prelude.NORMALISING
    assert evidence.NORMALISING is prelude.NORMALISING
    assert evidence.EXP_NAMES is prelude.EXP_NAMES
    assert evidence.MAX_NAMES is prelude.MAX_NAMES


SOFTMAX = """
__global__ void row_softmax(const float* in, float* out, int rows, int cols) {
  int row = blockIdx.x;
  if (row >= rows) return;
  float m = -3.402823466e+38f;
  for (int i = 0; i < cols; ++i) m = %s(m, in[row * cols + i]);
  float total = 0.0f;
  for (int i = 0; i < cols; ++i) total += %s(in[row * cols + i] - m);
  for (int i = 0; i < cols; ++i) out[row * cols + i] = %s(in[row * cols + i] - m) / total;
}
"""


@pytest.mark.parametrize("exp", sorted(prelude.EXP_NAMES))
@pytest.mark.parametrize("mx", sorted(prelude.MAX_NAMES))
def test_a_softmax_parses_in_every_spelling_of_exp_and_max(exp: str, mx: str):
    """The regression, stated over the whole cross product rather than one case.

    `__expf` is the one that mattered in practice, since it is what a fast-math
    kernel is written with. It parsed to `calls=['fmaxf']` and parse_errors=2.
    """
    kernels = parse_source(SOFTMAX % (mx, exp, exp), filename="t.cu")

    assert len(kernels) == 1
    facts = kernels[0]
    assert facts.parse_errors == 0, f"{exp}/{mx} did not resolve"
    assert exp in facts.calls, f"{exp} is declared but did not reach facts.calls"
    assert mx in facts.calls, f"{mx} is declared but did not reach facts.calls"


@pytest.mark.parametrize("exp", sorted(prelude.EXP_NAMES))
def test_a_softmax_is_proposed_in_every_spelling_of_exp(exp: str):
    """Parsing is not the point. Being substitutable is."""
    pytest.importorskip("torch")
    from hipbridge.verify import substitutions

    facts = parse_source(SOFTMAX % ("fmaxf", exp, exp), filename="t.cu")[0]
    proposal = substitutions.propose(facts, recognize(facts).pattern)

    assert proposal is not None, f"a softmax spelled with {exp} was refused"
    assert proposal.suite.name == "row_softmax"


def test_an_unresolved_construct_is_reported_rather_than_absorbed():
    """-ferror-limit=0 means parsing continues. The user has to be told it did."""
    source = SOFTMAX % ("fmaxf", "expf", "expf")
    source = source.replace("float m =", "cooperative_groups::thread_block b = xyz(); float m =")

    facts = parse_source(source, filename="t.cu")[0]
    assert facts.parse_errors > 0, "this fixture is meant to be unparseable in part"

    report = recognize(facts).report()
    assert "did not resolve" in report

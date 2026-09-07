"""RoPE and the affine norms: the rest of a transformer's inference path.

RoPE is the first kernel here that is not a reduction, which cost a new pattern
in the recognizer and exposed a parser bug. Both are pinned below.
"""

from __future__ import annotations

import pytest

from hipbridge import Pattern, parse_file, recognize
from hipbridge.verify import available

pytestmark = pytest.mark.skipif(not available(), reason="[verify] extra not installed")


def _facts(examples, name):
    return parse_file(examples / name)[0]


def test_a_loop_step_is_not_an_accumulation(examples):
    """`i += 256` advances a counter. `sum += x[i]` reduces a row.

    They are spelled with the same operator, and counting both classified the
    strided RoPE kernel as a serial reduction. Kernels whose real accumulator
    lives in shared memory escaped this only because the tree rule claims them
    first, which is luck rather than correctness.
    """
    tuned = _facts(examples, "rope_tuned.cu")
    assert tuned.scalar_accumulations == 0, "the loop step must not count as a reduction"
    assert recognize(tuned).pattern is Pattern.ROW_MAP

    # The discriminator still works where there IS an accumulation.
    softmax = _facts(examples, "row_softmax.cu")
    assert softmax.scalar_accumulations >= 1
    assert recognize(softmax).pattern is Pattern.REDUCE_SERIAL


def test_rope_is_a_map_not_a_reduction(examples):
    """Calling it elementwise or a reduction would both be lies."""
    for name in ("rope.cu", "rope_tuned.cu"):
        r = recognize(_facts(examples, name))
        assert r.pattern is Pattern.ROW_MAP, f"{name} is {r.pattern.value}"
        assert any("no accumulation" in line for line in r.rationale)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("rope.cu", "rope"),
        ("rope_tuned.cu", "rope"),
        ("rms_norm_affine.cu", "rms_norm_affine"),
        ("rms_norm_affine_tuned.cu", "rms_norm_affine"),
        ("rms_norm.cu", "rms_norm"),
        ("layer_norm_affine.cu", "layer_norm_affine"),
    ],
)
def test_each_kernel_reaches_its_own_substitute(examples, name, expected):
    """Affine and plain forms differ only in a weight, and must not be confused."""
    from hipbridge.verify import substitutions

    facts = _facts(examples, name)
    source = (examples / name).read_text(encoding="utf-8")
    proposal = substitutions.propose(source, facts, recognize(facts).pattern)

    assert proposal is not None, f"{name} should propose {expected}"
    assert proposal.name == expected


def test_rope_shapes_are_pairable():
    """An odd head dimension has no pairing, so it is not claimed as passing."""
    from hipbridge.verify.suites import ROPE

    shapes = list(ROPE.shapes())
    assert shapes, "the sweep must not be empty"
    assert all(c % 2 == 0 and c >= 2 for _, c in shapes), "odd widths cannot be rotated"
    assert any(c == 64 for _, c in shapes), "the wavefront width has to be in the sweep"


def test_rope_tables_are_half_as_wide():
    """One angle per channel pair, not per channel."""
    from hipbridge.verify.inputs import InputSpec
    from hipbridge.verify.suites import ROPE

    spec = InputSpec(shape=(8, 128))
    made = [o.build(spec) for o in ROPE.extras]

    assert [tuple(t.shape) for t in made] == [(8, 64), (8, 64)]
    assert [o.name for o in ROPE.extras] == ["cos_tab", "sin_tab"]

    # A cosine and a sine of the SAME angles, so the pair is a rotation. Two
    # independent adversarial tables are not one, and in fp16 the products
    # overflow to infinity, which gets blamed on the kernel.
    import torch

    cos, sin = made
    assert torch.allclose(cos * cos + sin * sin, torch.ones_like(cos), atol=1e-5)
    assert float(cos.abs().max()) <= 1.0 and float(sin.abs().max()) <= 1.0


def test_rope_matches_torch_on_cpu():
    """The Triton kernel is unproven here; the maths it claims to implement is not."""
    import torch

    from hipbridge.verify.suites import ROPE

    x = torch.randn(4, 8)
    cos = torch.randn(4, 4)
    sin = torch.randn(4, 4)

    got = ROPE.portable(x, cos, sin)
    want = torch.empty_like(x)
    for r in range(4):
        for i in range(4):
            x0, x1 = x[r, 2 * i], x[r, 2 * i + 1]
            want[r, 2 * i] = x0 * cos[r, i] - x1 * sin[r, i]
            want[r, 2 * i + 1] = x0 * sin[r, i] + x1 * cos[r, i]

    assert torch.allclose(got, want, atol=1e-6)


def test_a_tree_reduction_need_not_combine_with_a_compound_assignment(examples):
    """The competent implementations do not use `+=`, and were invisible.

    reduce.tree required a compound assignment into shared memory. An online
    softmax rescales a running sum onto a new maximum, and Welford merges
    partial moments; both are plain assignments. So the rule recognised the
    naive spelling of a tree reduction and missed the two kernels written by
    someone who knew what they were doing, which is the worst possible thing for
    a rule whose job is to identify competence.
    """
    for name in ("row_softmax_tuned.cu", "layer_norm_tuned.cu", "layer_norm_affine_tuned.cu"):
        facts = _facts(examples, name)
        result = recognize(facts)

        assert facts.shared_accumulations == 0, f"{name} should combine by assignment"
        assert facts.shared_in_halving_loop, f"{name} reduces in shared memory"
        assert result.pattern is Pattern.REDUCE_TREE, f"{name} is {result.pattern.value}"


def test_a_halving_loop_alone_is_not_a_tree_reduction(examples):
    """The stride has to touch the shared memory, or it proves nothing.

    Without that tie, any kernel with a shared buffer and a loop that halves
    something unrelated would be claimed as a reduction.
    """
    facts = _facts(examples, "rope_tuned.cu")

    assert not facts.shared_in_halving_loop
    assert recognize(facts).pattern is Pattern.ROW_MAP

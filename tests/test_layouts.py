"""Non-contiguous inputs, which nothing could previously build or survive.

Every kernel indexes as `row * row_stride + col * col_stride`. Until those
strides were passed the second factor was assumed to be 1, and nothing could
catch it because `generate()` only ever built fresh contiguous tensors: the
assumption in the kernels and the assumption in the test data agreed with each
other, so `softmax_rowwise(x.t())` returned wrong numbers with no error.

The kernels themselves run only on the MI300X. What is testable here is that the
sweep can now produce the layouts, that the values survive the rearrangement,
and that the harness actually runs them.
"""

from __future__ import annotations

import pytest

from hipbridge import verify

pytestmark = pytest.mark.skipif(not verify.available(), reason="[verify] extra not installed")


@pytest.fixture(scope="module")
def torch_():
    import torch

    return torch


def test_a_layout_changes_the_strides_and_nothing_else(torch_):
    """Same values, different arrangement. A kernel reading its strides properly
    cannot tell these apart; one that ignores them fails on the first case."""
    from hipbridge.verify import inputs

    base = inputs.generate(inputs.InputSpec((4, 8)))
    assert base.is_contiguous()

    for layout in inputs.NON_CONTIGUOUS:
        t = inputs.generate(inputs.InputSpec((4, 8), layout=layout))
        assert torch_.equal(t, base), layout
        assert not t.is_contiguous(), f"{layout} produced a contiguous tensor"


def test_padded_rows_are_the_case_that_matters_most(torch_):
    """A kernel written for rows padded to a hardware boundary has
    row_stride > n_cols. Proving it against packed data proves another program."""
    from hipbridge.verify import inputs

    t = inputs.generate(inputs.InputSpec((4, 8), layout=inputs.Layout.PADDED))
    assert t.stride(0) > t.shape[1], t.stride()
    assert t.stride(1) == 1, "padding changes the row stride, not the element stride"


def test_a_vector_has_no_transpose(torch_):
    """Weights are 1D, and `.t()` on a vector is not a layout, it is an error."""
    from hipbridge.verify import inputs

    spec = inputs.InputSpec((8,), layout=inputs.Layout.TRANSPOSED)
    assert inputs.generate(spec).stride() == (1,)

    sliced = inputs.generate(inputs.InputSpec((8,), layout=inputs.Layout.SLICED))
    assert sliced.stride() == (2,), "a sliced weight is what exercises gamma_stride"


def test_every_operand_gets_a_stride_not_just_the_primary(torch_):
    """gamma, beta, cos and sin are read with their own strides now, so the
    sweep has to hand them tensors that actually have one.

    The transform is the trap: torch.cos allocates a fresh contiguous tensor, so
    a padded table came back packed and the table strides went untested while
    appearing to be swept.
    """
    from hipbridge.verify import inputs, suites

    spec = inputs.InputSpec((4, 8), layout=inputs.Layout.SLICED)

    x, gamma, beta = suites.make_inputs(suites.LAYER_NORM_AFFINE, spec)
    assert x.stride(1) == 2
    assert gamma.stride(0) == 2 and beta.stride(0) == 2

    _, cos_tab, sin_tab = suites.make_inputs(suites.ROPE, spec)
    assert cos_tab.stride(1) == 2, "the cosine table came back packed"
    assert sin_tab.stride(1) == 2, "the sine table came back packed"


def test_the_layout_pass_survives_a_limit(torch_):
    """--limit is a cost control. Letting it drop the only cases that exercise
    strides would repeat the defect the multi-row sweep fix was written for."""
    from hipbridge.verify import inputs

    summary = verify.Harness(
        candidate=lambda t: torch_.softmax(t, dim=-1),
        reference=verify.TorchReference(lambda t: torch_.softmax(t, dim=-1)),
        oracle=lambda t: torch_.softmax(t.double(), dim=-1),
        layouts=inputs.NON_CONTIGUOUS,
        name="truncated",
    ).run([(1, 1), (2, 64)], limit=2)

    labels = [r.label for r in summary.results]
    assert len(labels) == 2 + len(inputs.NON_CONTIGUOUS), labels
    for layout in inputs.NON_CONTIGUOUS:
        assert any(layout.value in label for label in labels), (layout, labels)


def test_the_layout_pass_uses_the_widest_shape(torch_):
    """A stride mistake shows at width, not at (1, 1)."""
    from hipbridge.verify import inputs

    summary = verify.Harness(
        candidate=lambda t: torch_.softmax(t, dim=-1),
        reference=verify.TorchReference(lambda t: torch_.softmax(t, dim=-1)),
        oracle=lambda t: torch_.softmax(t.double(), dim=-1),
        layouts=(inputs.Layout.PADDED,),
        name="widest",
    ).run([(1, 1), (2, 512), (1000, 8)])

    padded = [r.label for r in summary.results if "padded" in r.label]
    assert padded == ["2x512/float32/normal/padded"], padded


def test_a_candidate_that_ignores_strides_is_caught(torch_):
    """The point of the whole change, demonstrated with a stand-in.

    This candidate reads the underlying storage as if it were packed, which is
    exactly what every kernel here did before the strides were threaded through.
    """
    from hipbridge.verify import inputs

    def packed_reader(x):
        # What `row * row_stride + col` computes when col_stride is assumed 1.
        flat = x.as_strided((x.shape[0], x.shape[1]), (x.stride(0), 1))
        return torch_.softmax(flat, dim=-1)

    summary = verify.Harness(
        candidate=packed_reader,
        reference=verify.TorchReference(lambda t: torch_.softmax(t, dim=-1)),
        oracle=lambda t: torch_.softmax(t.double(), dim=-1),
        layouts=(inputs.Layout.SLICED,),
        name="assumes packed",
    ).run([(4, 64)])

    assert not summary.ok, "a candidate ignoring its element stride passed"
    assert any("sliced" in r.label for r in summary.failures)

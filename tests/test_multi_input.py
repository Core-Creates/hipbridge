"""Kernels that take more than the tensor being transformed.

LayerNorm's gamma and beta are why this exists. Proving the normalisation alone
and then substituting an implementation that also multiplies by learned weights
would prove one program and ship another, so the harness has to carry the
weights through the candidate, the reference and the oracle alike.

These run on CPU with a torch reference, so the plumbing is covered without a
device. The device proof is a separate matter.
"""

from __future__ import annotations

import pytest

from hipbridge.verify import available

pytestmark = pytest.mark.skipif(not available(), reason="[verify] extra not installed")


@pytest.fixture(scope="module")
def torch_():
    import torch

    return torch


def _weights(offset):
    from hipbridge.verify.inputs import InputSpec

    def make(spec: InputSpec) -> InputSpec:
        return InputSpec(
            shape=(spec.shape[-1],), distribution=spec.distribution, seed=spec.seed + offset
        )

    return make


def test_extra_operands_reach_every_side(torch_):
    """Candidate, reference and oracle must all receive the weights."""
    from hipbridge import verify

    seen = {"candidate": 0, "reference": 0, "oracle": 0}

    def affine(x, gamma, beta):
        seen["candidate"] = 3
        return torch_.nn.functional.layer_norm(x, (x.shape[-1],), weight=gamma, bias=beta, eps=1e-5)

    def reference(x, gamma, beta):
        seen["reference"] = 3
        return torch_.nn.functional.layer_norm(x, (x.shape[-1],), weight=gamma, bias=beta, eps=1e-5)

    def oracle(x, gamma, beta):
        seen["oracle"] = 3
        d = x.double()
        mean = d.mean(dim=-1, keepdim=True)
        centred = d - mean
        var = (centred * centred).mean(dim=-1, keepdim=True)
        return centred * torch_.rsqrt(var + 1e-5) * gamma.double() + beta.double()

    summary = verify.Harness(
        candidate=affine,
        reference=reference,
        oracle=oracle,
        extras=(_weights(101), _weights(202)),
        name="affine",
        device="cpu",
    ).run([(4, 64)])

    assert summary.ok, str(summary)
    assert seen == {"candidate": 3, "reference": 3, "oracle": 3}


def test_a_kernel_that_ignores_its_weights_is_caught(torch_):
    """Dropping gamma and beta is the failure this coverage exists to catch.

    A candidate that normalises correctly and then ignores the weights looks
    perfect to a single-input harness, and is wrong everywhere gamma is not 1.
    """
    from hipbridge import verify

    def ignores_weights(x, gamma, beta):
        return torch_.nn.functional.layer_norm(x, (x.shape[-1],), eps=1e-5)

    def reference(x, gamma, beta):
        return torch_.nn.functional.layer_norm(x, (x.shape[-1],), weight=gamma, bias=beta, eps=1e-5)

    summary = verify.Harness(
        candidate=ignores_weights,
        reference=reference,
        extras=(_weights(101), _weights(202)),
        name="drops the affine step",
        device="cpu",
    ).run([(4, 64)])

    assert not summary.ok, "ignoring the weights must not pass"


def test_single_input_kernels_are_untouched(torch_):
    """The default is no extras, and that path must behave exactly as before."""
    from hipbridge import verify

    summary = verify.Harness(
        candidate=lambda t: torch_.softmax(t, dim=-1),
        reference=lambda t: torch_.softmax(t, dim=-1),
        name="softmax",
        device="cpu",
    ).run([(4, 64)])

    assert summary.ok, str(summary)


def test_weight_shapes_follow_the_case(torch_):
    """A sweep over shapes has to sweep the weights with it."""
    from hipbridge.verify.inputs import InputSpec
    from hipbridge.verify.suites import LAYER_NORM_AFFINE

    for cols in (63, 64, 1024):
        spec = InputSpec(shape=(8, cols))
        made = [m(spec) for m in LAYER_NORM_AFFINE.extras]
        assert [s.shape for s in made] == [(cols,), (cols,)]
        assert made[0].seed != made[1].seed, "gamma and beta must not be the same tensor twice"

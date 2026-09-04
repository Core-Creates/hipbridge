"""The harness must catch the failure modes that motivated this project.

Each test below is a kernel that is broken in a specific, realistic way. If the
harness passes any of them, the harness is worthless.
"""

from __future__ import annotations

import pytest

from hipbridge import verify

pytestmark = pytest.mark.skipif(not verify.available(), reason="[verify] extra not installed")


@pytest.fixture(scope="module")
def torch_():
    import torch

    return torch


@pytest.fixture
def reference(torch_):
    return verify.TorchReference(lambda x: torch_.softmax(x, dim=-1))


SMALL = [(4, 64), (1, 128), (64, 65)]


def _harness(candidate, reference, **kw):
    return verify.Harness(candidate=candidate, reference=reference, **kw)


def test_correct_implementation_passes(torch_, reference):
    h = _harness(lambda x: torch_.softmax(x, dim=-1), reference, name="correct")
    s = h.run(SMALL)
    assert s.ok, str(s)
    assert s.worst_ulp == 0


def test_catches_identity_kernel(torch_, reference):
    """The rocm-scribe failure: computation dropped, input copied through."""
    h = _harness(lambda x: x.clone(), reference, name="identity")
    s = h.run(SMALL)
    assert not s.ok
    assert any("IDENTITY" in f for r in s.failures for f in r.failures)


def test_catches_dropped_max_subtraction(torch_, reference):
    """Numerically unstable softmax. Correct on N(0,1), destroyed on large logits."""

    def unstable(x):
        e = torch_.exp(x)
        return e / e.sum(dim=-1, keepdim=True)

    h = _harness(unstable, reference, name="unstable")
    s = h.run(SMALL)
    assert not s.ok, "harness missed an overflow-prone softmax"
    # It must fail specifically on the large-magnitude inputs, not everywhere.
    assert any("large" in r.label for r in s.failures)


def test_passes_the_same_kernel_on_normal_inputs_only(torch_, reference):
    """Confirms the LARGE distribution is what catches it, not luck."""

    def unstable(x):
        e = torch_.exp(x)
        return e / e.sum(dim=-1, keepdim=True)

    h = _harness(
        unstable,
        reference,
        name="unstable-normal-only",
        distributions=(verify.Distribution.NORMAL,),
    )
    s = h.run([(4, 64)])
    assert s.ok, "unstable softmax should look fine on N(0,1); that is the point"


def test_catches_nondeterminism(torch_, reference):
    state = {"n": 0}

    def flaky(x):
        state["n"] += 1
        out = torch_.softmax(x, dim=-1)
        return out + (1e-6 if state["n"] % 2 == 0 else 0.0)

    h = _harness(flaky, reference, name="flaky")
    s = h.run([(4, 64)])
    assert not s.ok
    assert any("NONDETERMINISTIC" in f for r in s.failures for f in r.failures)


def test_deterministic_nan_is_not_called_nondeterministic(torch_, reference):
    """torch.equal reports NaN != NaN. A kernel that reliably returns NaN is
    still deterministic, and mislabelling it hides the real defect."""

    def always_nan(x):
        return torch_.full_like(x, float("nan"))

    s = _harness(always_nan, reference, name="nan").run([(4, 64)])
    assert not s.ok, "a NaN-producing kernel must still fail"
    flagged = [f for r in s.failures for f in r.failures if "NONDETERMINISTIC" in f]
    assert not flagged, f"deterministic NaN mislabelled: {flagged}"


def test_catches_wrong_axis(torch_, reference):
    """Reducing the wrong dimension: right shape, right magnitude, wrong answer."""
    h = _harness(lambda x: torch_.softmax(x, dim=0), reference, name="wrong-axis")
    s = h.run([(8, 16)])
    assert not s.ok


def test_raising_candidate_is_a_failure_not_a_crash(reference):
    def boom(x):
        raise RuntimeError("out of memory")

    s = _harness(boom, reference, name="raises").run([(4, 64)])
    assert not s.ok
    assert any("raised RuntimeError" in f for r in s.failures for f in r.failures)


def test_summary_reports_case_counts(torch_, reference):
    s = _harness(lambda x: torch_.softmax(x, dim=-1), reference).run(SMALL)
    expected = len(SMALL) * len(verify.DEFAULT_SWEEP)
    assert len(s.results) == expected
    assert "cases" in str(s)


def test_device_auto_resolves_to_something_concrete(torch_, reference):
    h = _harness(lambda x: torch_.softmax(x, dim=-1), reference)
    assert h.device in ("cuda", "cpu")
    assert h.device != "auto", "auto must be resolved in __post_init__"


def test_candidate_output_is_normalised_before_comparison(torch_, reference):
    """Regression: the harness ran a Triton candidate on an MI300X and every
    case failed with "Pointer argument cannot be accessed from Triton (cpu
    tensor?)". Inputs defaulted to CPU, which is fine for a torch candidate and
    impossible for a GPU kernel. Outputs must also be brought back to the host,
    since the reference returns a host tensor and mixing devices raises.
    """

    def needs_detach(x):
        # A candidate whose output carries grad and is non-contiguous, standing
        # in for one that returns a device tensor.
        y = torch_.softmax(x, dim=-1).clone().requires_grad_(True)
        return y.transpose(0, 1).transpose(0, 1)

    s = _harness(needs_detach, reference, name="detach").run([(4, 64)])
    assert s.ok, str(s)


def test_explicit_device_is_respected(torch_, reference):
    h = _harness(lambda x: torch_.softmax(x, dim=-1), reference, device="cpu")
    assert h.device == "cpu"

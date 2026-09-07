"""Accuracy arbitration against a high-precision oracle.

Measured on an RTX 4060 via WSL: examples/row_softmax.cu is 1.4x to 59.7x LESS
accurate than torch across every shape and distribution tried, because it
accumulates serially where torch reduces pairwise. Requiring a translated kernel
to match such an original within a ULP budget would reject the better
implementation, so accuracy is judged against float64 truth instead.
"""

from __future__ import annotations

import pytest

from hipbridge import verify

pytestmark = pytest.mark.skipif(not verify.available(), reason="[verify] extra not installed")


@pytest.fixture(scope="module")
def torch_():
    import torch

    return torch


def _softmax_case(torch_, n=1024, seed=3):
    x = verify.generate(
        verify.InputSpec((1, n), distribution=verify.Distribution.MIXED_SIGN, seed=seed)
    )
    truth = torch_.softmax(x.double(), dim=-1)
    return x, truth


def test_serial_summation_is_less_accurate_than_pairwise(torch_):
    """The finding that motivated oracle mode, reproduced without a GPU."""
    x, truth = _softmax_case(torch_, n=4096)

    # Pairwise (what torch and a tree-reduction Triton kernel do).
    pairwise = torch_.softmax(x, dim=-1)

    # Serial accumulation, mirroring the CUDA kernel's `for i: s += v`.
    e = torch_.exp(x - x.max(dim=-1, keepdim=True).values)
    total = torch_.zeros((), dtype=torch_.float32)
    for i in range(e.shape[-1]):
        total = total + e[0, i]
    serial = e / total

    arb = verify.arbitrate(serial, pairwise, truth)
    assert arb.candidate_err >= arb.reference_err, (
        f"serial summation should be no more accurate than pairwise; got {arb}"
    )


def test_more_accurate_candidate_is_accepted(torch_):
    """A better kernel must pass even though it diverges from the reference."""
    x, truth = _softmax_case(torch_)
    good = torch_.softmax(x, dim=-1)
    sloppy = (good.double() + 1e-6).float()  # deliberately worse reference

    arb = verify.arbitrate(good, sloppy, truth)
    assert arb.verdict in ("better", "equivalent"), str(arb)


def test_less_accurate_candidate_is_rejected(torch_):
    x, truth = _softmax_case(torch_)
    good = torch_.softmax(x, dim=-1)
    bad = (good.double() + 1e-3).float()

    arb = verify.arbitrate(bad, good, truth)
    assert arb.verdict == "worse", str(arb)


def test_sub_epsilon_ratios_are_not_failures(torch_):
    """Two implementations both accurate to a few ULP must not fail on ratio.

    Without an absolute floor, errors of 1e-8 and 4e-9 give a 2.4x ratio and a
    spurious rejection. That case was observed on real GPU output.
    """
    x, truth = _softmax_case(torch_)
    a = torch_.softmax(x, dim=-1)
    b = a.clone()
    # Perturb by one ULP: a genuine but meaningless difference.
    b.view(torch_.int32)[0, 0] += 1

    arb = verify.arbitrate(b, a, truth)
    assert arb.verdict != "worse", f"one-ULP perturbation rejected: {arb}"


def test_candidate_at_the_floor_beats_a_reference_that_is_not(torch_):
    """The floor is a noise guard, not a cap on how well a candidate can score.

    Observed on an MI300X: the tuned Triton softmax sat at or below the floor on
    every finite distribution while the serial original was up to 78x further
    from float64 truth, and every case was reported "equivalent" because the
    floor test looked at the candidate alone. A win that large is not noise.
    """
    x, truth = _softmax_case(torch_)
    good = torch_.softmax(x, dim=-1)
    sloppy = (good.double() + 1e-6).float()

    arb = verify.arbitrate(good, sloppy, truth)
    assert arb.candidate_err < arb.reference_err, str(arb)
    assert arb.verdict == "better", f"a clear win reported as {arb.verdict}: {arb}"


def test_both_sides_at_the_floor_stay_equivalent(torch_):
    """The noise guard itself must survive: near-exact pairs are not ranked."""
    x, truth = _softmax_case(torch_)
    a = torch_.softmax(x, dim=-1)
    b = a.clone()
    b.view(torch_.int32)[0, 0] += 1  # one ULP apart, both essentially exact

    arb = verify.arbitrate(b, a, truth)
    assert arb.verdict == "equivalent", f"noise ranked as {arb.verdict}: {arb}"


def test_nan_from_both_implementations_does_not_blame_the_candidate(torch_):
    """NaN cannot be ordered, so it has to be decided before the comparisons.

    Every `<=` against NaN answers False, which used to fall through to "worse"
    and fail a candidate for a NaN the reference produced identically.
    """
    x, truth = _softmax_case(torch_)
    nan = torch_.full_like(truth, float("nan")).float()

    both = verify.arbitrate(nan, nan, truth)
    assert both.verdict == "equivalent", f"shared NaN blamed on the candidate: {both}"

    good = torch_.softmax(x, dim=-1)
    only_candidate = verify.arbitrate(nan, good, truth)
    assert only_candidate.verdict == "worse", str(only_candidate)

    only_reference = verify.arbitrate(good, nan, truth)
    assert only_reference.verdict == "better", str(only_reference)


def test_summary_reports_the_margin_not_just_the_verdict(torch_):
    """A verdict count alone cannot separate a 1.02x edge from a 78x one."""
    oracle = lambda t: torch_.softmax(t.double(), dim=-1)  # noqa: E731
    sloppy = verify.TorchReference(lambda t: (torch_.softmax(t.double(), dim=-1) + 1e-6).float())

    summary = verify.Harness(
        candidate=lambda t: torch_.softmax(t, dim=-1),
        reference=sloppy,
        oracle=oracle,
        name="margin",
    ).run([(4, 64)])

    assert summary.accuracy_gain and summary.accuracy_gain > 1.5, str(summary)
    assert "closer to float64" in str(summary), str(summary)


def test_identity_kernel_is_still_rejected_in_oracle_mode(torch_):
    """Oracle mode must not become a loophole for a dropped computation."""
    x, truth = _softmax_case(torch_)
    good = torch_.softmax(x, dim=-1)

    arb = verify.arbitrate(x, good, truth)  # candidate = the input itself
    assert arb.verdict == "worse", str(arb)


def test_harness_oracle_mode_accepts_correct_and_rejects_identity(torch_):
    oracle = lambda t: torch_.softmax(t.double(), dim=-1)  # noqa: E731
    ref = verify.TorchReference(lambda t: torch_.softmax(t, dim=-1))
    shapes = [(4, 64), (1, 257)]

    ok = verify.Harness(
        candidate=lambda t: torch_.softmax(t, dim=-1),
        reference=ref,
        oracle=oracle,
        name="correct",
    ).run(shapes)
    assert ok.ok, str(ok)

    bad = verify.Harness(
        candidate=lambda t: t.clone(), reference=ref, oracle=oracle, name="identity"
    ).run(shapes)
    assert not bad.ok
    assert any("LESS ACCURATE" in f for r in bad.failures for f in r.failures)


def test_an_exact_tie_is_equivalent_not_better(torch_):
    """Identical implementations cannot be closer to the truth than each other.

    The rule was `ratio <= 1.0`, and a tie gives exactly 1.0, so every tie above
    the noise floor scored as a win. Visible on hardware: the RoPE run reported
    `worst ulp=0` and `better=12` on the same line, which cannot both be true of
    output that is bitwise identical to the original.
    """
    x, truth = _softmax_case(torch_)
    same = (torch_.softmax(x.double(), dim=-1) + 1e-3).float()  # well above the floor

    arb = verify.arbitrate(same, same, truth)
    assert arb.ratio == 1.0
    assert arb.verdict == "equivalent", f"a tie scored as {arb.verdict}"


def test_strictly_closer_is_still_better(torch_):
    """Tightening the tie must not cost a real win."""
    x, truth = _softmax_case(torch_)
    good = torch_.softmax(x, dim=-1)
    sloppy = (good.double() + 1e-6).float()

    arb = verify.arbitrate(good, sloppy, truth)
    assert arb.ratio < 1.0
    assert arb.verdict == "better", str(arb)

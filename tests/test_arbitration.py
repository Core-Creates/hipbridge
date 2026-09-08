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


def test_a_nan_in_the_oracle_does_not_excuse_the_finite_elements(torch_):
    """The blind pass: one NaN used to decide the whole case.

    Both errors were a max over the entire tensor, so a single NaN in the truth
    made both of them NaN, and the branch catching "NaN on both sides" returned
    equivalent for everything else in the tensor. In oracle mode the harness has
    already dropped the ULP and identity checks, so nothing else was looking.

    The two calls below differ in one element and in nothing else.
    """
    nan = float("nan")
    garbage = torch_.tensor([[nan, 999.0, -42.0, 0.0]])
    honest = torch_.tensor([[nan, 1.0, 2.0, 3.0]])
    truth = torch_.tensor([[nan, 1.0, 2.0, 3.0]], dtype=torch_.float64)

    with_nan = verify.arbitrate(garbage, honest, truth)
    without_nan = verify.arbitrate(
        torch_.tensor([[0.0, 999.0, -42.0, 0.0]]),
        torch_.tensor([[0.0, 1.0, 2.0, 3.0]]),
        torch_.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch_.float64),
    )

    assert with_nan.verdict == "worse", str(with_nan)
    assert with_nan.verdict == without_nan.verdict
    assert with_nan.candidate_err == pytest.approx(without_nan.candidate_err)


def test_the_non_finite_pattern_is_judged_before_the_magnitudes(torch_):
    """Producing a number where the oracle produces NaN answers a different question."""
    nan = float("nan")
    truth = torch_.tensor([[nan, 1.0, 2.0, 3.0]], dtype=torch_.float64)
    agrees = torch_.tensor([[nan, 1.0, 2.0, 3.0]])
    invents = torch_.tensor([[5.0, 1.0, 2.0, 3.0]])

    assert verify.arbitrate(invents, agrees, truth).verdict == "worse"
    assert verify.arbitrate(agrees, invents, truth).verdict == "better"
    assert verify.arbitrate(agrees, agrees, truth).verdict == "equivalent"


def test_the_noise_floor_is_taken_over_the_finite_part(torch_):
    """A NaN anywhere made the scale NaN, which disabled the floor as well.

    Every `<=` against a NaN floor answers False, so two implementations that
    were both essentially exact stopped being ranked as noise and fell through
    to the ratio.
    """
    nan = float("nan")
    truth = torch_.tensor([[nan, 1.0, 2.0, 3.0]], dtype=torch_.float64)
    a = torch_.tensor([[nan, 1.0, 2.0, 3.0]])
    b = a.clone()
    b[0, 1] = float(torch_.nextafter(b[0, 1], torch_.tensor(2.0)))

    assert verify.arbitrate(b, a, truth).verdict == "equivalent"


def test_nan_for_a_number_fails_even_when_the_reference_agrees(torch_):
    """Ranking is relative, so something else has to fail a shared failure.

    Two implementations that both return NaN where the oracle returns a number
    rank as equivalent, which is true and useless. nonfinite_where_finite is the
    gate that fails the case, and it survives oracle mode where the ULP and
    identity checks do not.
    """
    oracle = lambda t: torch_.softmax(t.double(), dim=-1)  # noqa: E731
    broken = verify.TorchReference(lambda t: torch_.full_like(t, float("nan")))

    summary = verify.Harness(
        candidate=lambda t: torch_.full_like(t, float("nan")),
        reference=broken,
        oracle=oracle,
        name="both broken",
        # This case deliberately measures a broken reference, which is what the
        # sanity probe exists to refuse. Opting out is why probe_cases is a
        # field: the probe would reject this reference first and the gate under
        # test would never be reached.
        probe_cases=0,
    ).run([(4, 64)])

    assert not summary.ok, str(summary)
    assert "NOT FINITE" in str(summary), str(summary)


def test_a_correct_candidate_still_passes_the_finiteness_gate(torch_):
    """The gate must not fire on the ordinary case it sits in front of."""
    oracle = lambda t: torch_.softmax(t.double(), dim=-1)  # noqa: E731
    ref = verify.TorchReference(lambda t: torch_.softmax(t, dim=-1))

    summary = verify.Harness(
        candidate=lambda t: torch_.softmax(t, dim=-1),
        reference=ref,
        oracle=oracle,
        name="ordinary",
    ).run([(4, 64), (1, 257)])

    assert summary.ok, str(summary)
    assert "NOT FINITE" not in str(summary), str(summary)


def test_a_wrong_row_cannot_hide_behind_a_larger_one(torch_):
    """The floor followed the tensor's largest output, not each row's own.

    In bfloat16 that floor is 6.25% of the largest value anywhere, so a row of
    magnitude 1 beside a row of magnitude 100 could be wrong in every element
    and still sit under it. Row-wise kernels are precisely where magnitudes
    differ between rows: an attention row that masks to almost nothing beside
    one that does not is the ordinary case.
    """
    truth = torch_.tensor([[100.0, 100.0], [1.0, 1.0]], dtype=torch_.float64)
    reference = truth.to(torch_.bfloat16)
    candidate = torch_.tensor([[100.0, 100.0], [-5.0, 7.0]], dtype=torch_.bfloat16)

    assert verify.arbitrate(candidate, reference, truth).verdict == "worse"
    # The same tensor without the large row was never in doubt; it is the
    # neighbour that used to buy the wrong row its pass.
    alone = verify.arbitrate(candidate[1:], reference[1:], truth[1:])
    assert alone.verdict == "worse", str(alone)


def test_matching_rows_are_still_noise_in_every_precision(torch_):
    """The per-row floor must not start failing implementations that agree."""
    truth = torch_.tensor([[100.0, 100.0], [1.0, 1.0], [0.0, 0.0]], dtype=torch_.float64)
    for dtype in (torch_.float32, torch_.float16, torch_.bfloat16):
        a = truth.to(dtype)
        arb = verify.arbitrate(a.clone(), a, truth)
        assert arb.verdict == "equivalent", (dtype, str(arb))


def test_a_row_wise_win_is_still_reported_as_one(torch_):
    """Per-row strictness must not suppress a candidate that is better everywhere."""
    truth = torch_.tensor([[1.0, 2.0], [10.0, 20.0]], dtype=torch_.float64)
    close = torch_.tensor([[1.001, 2.001], [10.01, 20.01]])
    far = torch_.tensor([[1.1, 2.1], [11.0, 21.0]])

    arb = verify.arbitrate(close, far, truth)
    assert arb.verdict == "better", str(arb)
    assert arb.ratio < 1.0


def test_a_row_the_original_got_exactly_right_does_not_fail_the_case(torch_):
    """The per-row guard's first version failed a candidate that was 12x closer.

    Measured on an MI300X: layer_norm at 64x65 and 1000x128 on monotonic input.
    The naive kernel lands nearly exact on 17 of those 64 rows, so the ratio
    between the two implementations there is 36x, while the candidate's error is
    six millionths of a percent of the row. Failing that is the
    match-the-original's-rounding test this project exists to argue against.

    A row now has to be wrong on its own terms, not merely worse than a
    reference that happened to be exact.
    """
    truth = torch_.tensor(
        [[1.4, -1.4], [1.4, -1.4], [1.4, -1.4], [1.4, -1.4]], dtype=torch_.float64
    )
    # Exact on the first two rows, badly off on the last two.
    reference = truth.clone()
    reference[2:] += 1.028e-4
    # Slightly off everywhere, and far closer overall.
    candidate = truth + 8.779e-6

    arb = verify.arbitrate(candidate.float(), reference.float(), truth)
    assert arb.verdict == "better", str(arb)
    assert arb.ratio < 1.0

    # The guard still fires when a row is wrong on its own terms.
    broken = candidate.clone()
    broken[0] = 0.0  # a whole row, 100% of its own magnitude
    assert verify.arbitrate(broken.float(), reference.float(), truth).verdict == "worse"

"""The reference has to be worth comparing against before anything is compared.

Oracle mode passes the candidate when it is closer to the truth than the
reference is. If the reference is garbage the candidate is trivially closer and
the run reports a proof: measured once at 3.9e75x "better", from a tuned kernel
launched at the wrong block size that read uninitialised shared memory.

The check for that lived in `port` and nowhere else, so `verify`, `bench` and
`synth` ran without it, `synth` being the one that executes generated code.
"""

from __future__ import annotations

import pytest

from hipbridge import verify

pytestmark = pytest.mark.skipif(not verify.available(), reason="[verify] extra not installed")


@pytest.fixture(scope="module")
def torch_():
    import torch

    return torch


def _oracle(torch_):
    return lambda t: torch_.softmax(t.double(), dim=-1)


def test_a_sane_reference_passes_and_the_sweep_runs(torch_):
    summary = verify.Harness(
        candidate=lambda t: torch_.softmax(t, dim=-1),
        reference=verify.TorchReference(lambda t: torch_.softmax(t, dim=-1)),
        oracle=_oracle(torch_),
        name="sane",
    ).run([(4, 64), (1, 257)])

    assert summary.probe_failure is None, summary.probe_failure
    assert summary.ok and summary.results


def test_a_garbage_reference_is_refused_before_the_sweep(torch_):
    """And nothing is reported as proved against it."""
    summary = verify.Harness(
        candidate=lambda t: torch_.softmax(t, dim=-1),
        reference=verify.TorchReference(lambda t: torch_.full_like(t, 7.0)),
        oracle=_oracle(torch_),
        name="garbage",
    ).run([(4, 64), (1, 257)])

    assert summary.probe_failure is not None
    assert not summary.ok
    assert summary.results == [], "the sweep ran against a reference already known bad"
    assert "REFERENCE IS NOT SANE" in str(summary)


def test_a_reference_that_raises_is_a_diagnosis_not_a_traceback(torch_):
    """A compile error, a launch fault, a non-__global__ kernel: all arrived raw."""

    def explodes(_t):
        raise RuntimeError("hipcc: error: no such file or directory")

    summary = verify.Harness(
        candidate=lambda t: torch_.softmax(t, dim=-1),
        reference=verify.TorchReference(explodes),
        oracle=_oracle(torch_),
        name="explodes",
    ).run([(4, 64)])

    assert summary.probe_failure is not None
    assert "could not be run" in summary.probe_failure
    assert "hipcc" in summary.probe_failure


def test_the_probe_is_relative_so_small_outputs_are_judged_fairly(torch_):
    """An absolute 1e-3 was meaningless for outputs that are themselves ~4e-3.

    A 256-wide softmax sums to 1, so its elements average 1/256. A reference 20%
    wrong on every element misses by about 8e-4 absolute and passed the old
    threshold with room to spare.
    """
    wrong_by_20_percent = verify.TorchReference(lambda t: torch_.softmax(t, dim=-1) * 1.2)

    summary = verify.Harness(
        candidate=lambda t: torch_.softmax(t, dim=-1),
        reference=wrong_by_20_percent,
        oracle=_oracle(torch_),
        name="20 percent out",
    ).run([(4, 256)])

    assert summary.probe_failure is not None, "a 20% error passed as sane"


def test_half_precision_is_not_failed_for_being_half_precision(torch_):
    """The threshold follows the precision, or bf16 fails on its own resolution."""
    for dtype in (torch_.float16, torch_.bfloat16):
        summary = verify.Harness(
            candidate=lambda t: torch_.softmax(t, dim=-1),
            reference=verify.TorchReference(lambda t: torch_.softmax(t, dim=-1)),
            oracle=_oracle(torch_),
            dtypes=(dtype,),
            name=f"half {dtype}",
        ).run([(4, 64)])

        assert summary.probe_failure is None, (dtype, summary.probe_failure)
        assert summary.ok


def test_the_probe_spans_shapes_distributions_and_precisions(torch_):
    """It used to be one shape, one distribution, float32, which is three blind spots."""
    harness = verify.Harness(
        candidate=lambda t: t,
        reference=verify.TorchReference(lambda t: t),
        oracle=_oracle(torch_),
        dtypes=(torch_.float32, torch_.float16),
    )
    specs = harness._probe_specs([(1, 1), (2, 4096), (1000, 128)], harness.dtypes)

    assert len({s.shape for s in specs}) > 1, "one shape again"
    assert len({s.dtype for s in specs}) > 1, "one precision again"
    assert (2, 4096) in {s.shape for s in specs}, "the widest row is not sampled"
    assert len(specs) <= harness.probe_cases


def test_a_caller_can_opt_out(torch_):
    """probe_cases=0 for a test that is deliberately measuring a broken reference."""
    summary = verify.Harness(
        candidate=lambda t: torch_.softmax(t, dim=-1),
        reference=verify.TorchReference(lambda t: torch_.full_like(t, 7.0)),
        oracle=_oracle(torch_),
        probe_cases=0,
        name="opted out",
    ).run([(4, 64)])

    assert summary.probe_failure is None
    assert summary.results, "the sweep should have run"

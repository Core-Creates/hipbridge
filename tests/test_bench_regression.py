"""The comparator has to fire on a regression and stay quiet on a wobble.

A check that cries wolf gets switched off, so the interesting test is not that
it catches a 2x slowdown - anything catches that - but that it ignores a move
the measurement cannot distinguish from noise.

No torch, no Triton, no device. It reads two reports.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _comparator():
    spec = importlib.util.spec_from_file_location(
        "cmp_bench", REPO / "scripts" / "compare-bench.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report(candidate_median: float, low: float, high: float) -> str:
    """One kernel, one dtype, one shape, in the committed report's layout."""
    return (
        "# hipbridge benchmark\n\n"
        "## row_softmax (float32)\n\n"
        "```\n"
        "       shape             original (us)          tuned HIP (us)"
        "              torch (us)          candidate (us)\n"
        "   4096x4096    2429.8 [2429.4-2430.3]        52.3 [52.1-53.1]"
        "        40.1 [39.4-40.7]"
        f"        {candidate_median} [{low}-{high}]        74.6x         1.6x         1.2x\n"
        "```\n"
    )


def test_the_committed_report_parses():
    """Against the real file, so the layout cannot drift away from the parser."""
    report = REPO / "results" / "bench-hipcc-gfx942.md"
    if not report.is_file():
        pytest.skip("no benchmark committed")

    rows = _comparator().parse(report.read_text(encoding="utf-8"))

    assert len(rows) > 20, f"parsed only {len(rows)} rows from the committed benchmark"
    assert ("row_softmax", "float32", "4096x4096") in rows


def test_a_real_slowdown_is_caught():
    m = _comparator()
    before = m.parse(_report(32.6, 32.4, 32.7))
    after = m.parse(_report(48.0, 47.5, 48.4))

    found = m.regressions(before, after, 0.15)

    assert any("SLOWER" in f for f in found), found


def test_a_wobble_inside_the_noise_band_is_not():
    """The bands still overlap, so the two numbers have not been shown to differ.

    This is the case that decides whether anyone leaves the check switched on.
    """
    m = _comparator()
    before = m.parse(_report(32.6, 30.0, 40.0))
    after = m.parse(_report(38.0, 33.0, 42.0))  # +17% median, bands overlap

    found = m.regressions(before, after, 0.15)

    assert not any("SLOWER" in f for f in found), found
    assert any("noted" in f for f in found), "an overlapping move should still be visible"


def test_a_move_under_the_threshold_is_silent():
    m = _comparator()
    before = m.parse(_report(32.6, 32.4, 32.7))
    after = m.parse(_report(35.0, 34.9, 35.1))  # +7%, bands clear but small

    assert m.regressions(before, after, 0.15) == []


def test_a_kernel_that_vanished_is_not_a_regression():
    """It is a change worth naming, but the exit code belongs to slowdowns."""
    m = _comparator()
    before = m.parse(_report(32.6, 32.4, 32.7))

    assert m.regressions(before, {}, 0.15) == []

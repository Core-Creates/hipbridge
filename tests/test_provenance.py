"""Reports have to say where they came from, and survive the box that made them.

Every measurement in this project's README arrived by a human copying it out of
a terminal, and the rented MI300X that produced them went down twice, taking its
untracked report with it each time. These tests pin the parts that make a
committed result checkable later.
"""

from __future__ import annotations

import pytest

from hipbridge.verify import available

pytestmark = pytest.mark.skipif(not available(), reason="[verify] extra not installed")


def test_default_paths_are_deterministic():
    """A re-run should be a diff, not a new file nobody compares."""
    from hipbridge.verify import provenance

    a = provenance.default_path("verify", "hipcc", "gfx942")
    b = provenance.default_path("verify", "hipcc", "gfx942")
    assert a == b
    assert a.replace("\\", "/") == "results/verify-hipcc-gfx942.md"
    assert provenance.default_path("bench", "nvcc", "").replace("\\", "/") == (
        "results/bench-nvcc-default.md"
    )


def test_the_header_names_the_machine_and_the_commit():
    from hipbridge.verify import provenance

    head = provenance.header("t", "hipcc", "gfx942")
    for field in ("generated", "hipbridge", "host", "device", "toolchain", "arch", "torch"):
        assert f"| {field} |" in head, f"{field} missing from the provenance header"
    assert "gfx942" in head


def test_provenance_never_raises_on_a_machine_without_the_tools():
    """A report that fails to write because a GPU name was unreadable is a bad trade."""
    from hipbridge.verify import provenance

    assert provenance.toolchain_version("definitely-not-a-real-compiler") == "unknown"
    assert isinstance(provenance.device(), str)
    assert isinstance(provenance.commit(), str)


def test_write_creates_the_directory_and_keeps_the_body(tmp_path):
    from hipbridge.verify import provenance

    target = tmp_path / "nested" / "report.md"
    written = provenance.write(target, "a title", "hipcc", "gfx942", "PASS  something")
    text = written.read_text(encoding="utf-8")

    assert written.exists()
    assert text.startswith("# a title")
    assert "PASS  something" in text


def test_absolute_error_travels_with_the_ulp_count():
    """1.7 billion ULP on a centred output is arithmetic, not a defect.

    LayerNorm outputs straddle zero, where +1e-9 and -1e-9 are a hair apart in
    magnitude and astronomically far apart on the ULP integer line. Reporting
    only the ULP figure in a committed file would read as a catastrophe, so the
    magnitude that produced it is printed beside it.
    """
    from hipbridge.verify.harness import CaseResult, Summary

    s = Summary("norm", results=[CaseResult("case", True, max_ulp=1776828265, max_abs=3.91e-05)])
    text = str(s)

    assert "worst ulp=1776828265" in text
    assert "max abs 3.910e-05" in text
    assert s.worst_abs == pytest.approx(3.91e-05)


def test_results_do_not_count_against_their_own_provenance(tmp_path, monkeypatch):
    """Writing a report must not stamp the report -dirty.

    The first run of this feature produced files headed `538007a-dirty` from a
    pristine checkout, because the untracked results/ directory it had just
    created was itself the only modification. A marker that is always on carries
    no information.
    """
    from hipbridge.verify import provenance

    def fake_run(cmd):
        if cmd[:2] == ["git", "rev-parse"]:
            return "abc1234"
        if cmd[:2] == ["git", "status"]:
            return "?? results/verify-hipcc-gfx942.md\n?? results/bench-hipcc-gfx942.md"
        return ""

    monkeypatch.setattr(provenance, "_run", fake_run)
    assert provenance.commit() == "abc1234", "results/ must not mark the tree dirty"

    def fake_run_real_change(cmd):
        if cmd[:2] == ["git", "rev-parse"]:
            return "abc1234"
        if cmd[:2] == ["git", "status"]:
            return " M src/hipbridge/verify/bench.py\n?? results/verify.md"
        return ""

    monkeypatch.setattr(provenance, "_run", fake_run_real_change)
    assert provenance.commit() == "abc1234-dirty", "a real source change must still show"

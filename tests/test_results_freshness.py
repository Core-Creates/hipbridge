"""Committed results have to stay honest about what they measured.

A report in `results/` carries the commit it was produced at, which is what
makes it checkable. Nothing checked it, and the drift arrived exactly as
predicted: the files sat at a commit describing three suites while the README
beside them described six, and the only reason anyone noticed was that someone
asked.

A report goes stale when the CODE it measured changes, not when any commit
lands. A README edit does not invalidate a number; a new kernel does.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hipbridge.verify import available

pytestmark = pytest.mark.skipif(not available(), reason="[verify] extra not installed")

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"


def _reports() -> list[Path]:
    return sorted(RESULTS.glob("*.md")) if RESULTS.is_dir() else []


@pytest.mark.parametrize("path", _reports(), ids=lambda p: p.name)
def test_a_committed_report_names_the_commit_it_measured(path: Path):
    """Without it, "84/84" describes an unknown program."""
    from hipbridge.verify import provenance

    stamped = provenance.parse_commit(path.read_text(encoding="utf-8"))

    assert stamped, f"{path.name} has no commit in its provenance header"


@pytest.mark.parametrize("path", _reports(), ids=lambda p: p.name)
def test_a_committed_report_is_not_stale(path: Path):
    """Fails when measured code changed after the report was produced.

    Skips rather than fails when the question is unanswerable. CI checks out a
    single commit by default, so ancestry for anything older cannot be resolved
    there, and a freshness check that fails on a shallow clone would be switched
    off inside a week.
    """
    from hipbridge.verify import provenance

    verdict, why = provenance.staleness(path.read_text(encoding="utf-8"))

    if verdict == "unknown":
        pytest.skip(f"{path.name}: {why}")
    assert verdict == "fresh", (
        f"{path.name} is stale: {why}. Re-run the measurement and commit it, "
        f"or delete the report rather than leaving it to read as current."
    )


def test_the_verify_report_covers_every_suite():
    """A report from a three-suite era should be visibly incomplete.

    Freshness by commit is necessary and not sufficient: a report produced at
    the right commit could still have run half the suites, and would look
    authoritative either way.
    """
    from hipbridge.verify.suites import BUILTIN

    report = RESULTS / "verify-hipcc-gfx942.md"
    if not report.is_file():
        pytest.skip("no verification report committed")

    text = report.read_text(encoding="utf-8")
    missing = [s.name for s in BUILTIN if s.name not in text]

    assert not missing, f"the committed verification does not mention: {', '.join(missing)}"


def test_staleness_reads_the_commit_out_of_a_header():
    """The parser, without depending on what happens to be committed."""
    from hipbridge.verify import provenance

    header = provenance.header("t", "hipcc", "gfx942")
    assert provenance.parse_commit(header), "a freshly built header must be readable"

    assert provenance.parse_commit("nothing to see here") is None
    assert provenance.parse_commit("| hipbridge | 0.1 at commit `abc1234` |") == "abc1234"
    # A dirty tree still names its commit; the marker is not part of the sha.
    assert provenance.parse_commit("| hipbridge | 0.1 at commit `abc1234-dirty` |") == "abc1234"


def test_an_unreadable_history_is_unknown_rather_than_stale(monkeypatch):
    """Unanswerable is not the same as wrong, and must not fail a build."""
    from hipbridge.verify import provenance

    monkeypatch.setattr(provenance, "_run", lambda cmd: "")
    verdict, why = provenance.staleness("| hipbridge | 0.1 at commit `abc1234` |")

    assert verdict == "unknown"
    assert "unavailable" in why or "not in this clone" in why

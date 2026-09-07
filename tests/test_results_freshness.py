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


def test_freshness_survives_a_rebase():
    """Ancestry was the wrong question, and it failed on this repo's own workflow.

    Every merge here is a rebase merge, which rewrites the commit a report was
    produced at, so a report generated on a feature branch named a commit that
    never landed on main. The check called that stale while the code it measured
    had not changed at all. Content answers the question ancestry could not.
    """
    from hipbridge.verify import provenance

    header = provenance.header("t", "hipcc", "gfx942")
    verdict, why = provenance.staleness(header)

    assert verdict == "fresh", why
    assert provenance.parse_code_digest(header) == provenance.code_digest()


def test_a_report_without_a_digest_is_unknown_not_stale():
    """Reports written before digests existed cannot be judged, only re-run."""
    from hipbridge.verify import provenance

    old_style = "| hipbridge | 0.1 at commit `abc1234` |"
    verdict, why = provenance.staleness(old_style)

    assert verdict == "unknown"
    assert "predates" in why


def test_a_changed_kernel_makes_a_report_stale(tmp_path):
    """The whole point: editing measured code has to invalidate the numbers."""
    from hipbridge.verify import provenance

    (tmp_path / "examples").mkdir()
    kernel = tmp_path / "examples" / "k.cu"
    kernel.write_text("__global__ void k() {}", encoding="utf-8")
    before = provenance.code_digest(tmp_path)

    kernel.write_text("__global__ void k() { int x = 1; }", encoding="utf-8")
    after = provenance.code_digest(tmp_path)

    assert before != after, "a changed kernel must change the digest"


def test_staleness_reads_the_commit_out_of_a_header():
    """The parser, without depending on what happens to be committed."""
    from hipbridge.verify import provenance

    header = provenance.header("t", "hipcc", "gfx942")
    assert provenance.parse_commit(header), "a freshly built header must be readable"

    assert provenance.parse_commit("nothing to see here") is None
    assert provenance.parse_commit("| hipbridge | 0.1 at commit `abc1234` |") == "abc1234"
    # A dirty tree still names its commit; the marker is not part of the sha.
    assert provenance.parse_commit("| hipbridge | 0.1 at commit `abc1234-dirty` |") == "abc1234"


def test_a_report_missing_its_commit_is_unknown():
    """A report that does not say where it came from cannot be judged at all.

    This used to test git being unavailable, which no longer matters: freshness
    reads the working tree, so a shallow clone, a missing git and an offline box
    all answer the same question correctly.
    """
    from hipbridge.verify import provenance

    verdict, why = provenance.staleness("no header at all")

    assert verdict == "unknown"
    assert "does not name the commit" in why

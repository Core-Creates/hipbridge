"""Reports have to say where they came from, and survive the box that made them.

Every measurement in this project's README arrived by a human copying it out of
a terminal, and the rented MI300X that produced them went down twice, taking its
untracked report with it each time. These tests pin the parts that make a
committed result checkable later.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hipbridge.verify import available

REPO = Path(__file__).resolve().parents[1]

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


# --- what the digest is allowed to notice ---------------------------------


def _tree(tmp_path):
    import shutil

    for name in ("src", "examples"):
        shutil.copytree(REPO / name, tmp_path / name)
    return tmp_path


def test_prose_does_not_invalidate_a_measurement(tmp_path):
    """Comments and docstrings never execute, so they cannot move a number.

    They were 42% of verify/ by volume, and hashing them meant every
    explanatory comment cost a metered GPU run to restore green: a tax on the
    habit this project most wants to keep.
    """
    from hipbridge.verify import provenance

    tree = _tree(tmp_path)
    before = provenance.code_digest(tree)

    target = tree / "src" / "hipbridge" / "verify" / "harness.py"
    source = target.read_text(encoding="utf-8")
    target.write_text(
        "# an explanatory comment nobody will ever execute\n"
        + source.replace(
            "Run a candidate against a reference", "Run a CANDIDATE against a REFERENCE"
        ),
        encoding="utf-8",
    )

    assert provenance.code_digest(tree) == before


def test_a_change_that_can_move_a_number_still_invalidates(tmp_path):
    """The point is to narrow what counts, not to stop counting."""
    from hipbridge.verify import provenance

    tree = _tree(tmp_path)
    before = provenance.code_digest(tree)

    target = tree / "src" / "hipbridge" / "verify" / "harness.py"
    target.write_text(
        target.read_text(encoding="utf-8").replace("probe_cases: int = 6", "probe_cases: int = 12"),
        encoding="utf-8",
    )

    assert provenance.code_digest(tree) != before


def test_a_cu_comment_does_not_invalidate_but_the_kernel_does(tmp_path):
    """rms_norm.cu carries ten lines explaining why it is deliberately naive."""
    from hipbridge.verify import provenance

    tree = _tree(tmp_path)
    before = provenance.code_digest(tree)

    target = tree / "examples" / "rms_norm.cu"
    source = target.read_text(encoding="utf-8")
    target.write_text("// a note about why this is the naive form\n" + source, encoding="utf-8")
    assert provenance.code_digest(tree) == before

    target.write_text(source.replace("1e-5f", "1e-6f"), encoding="utf-8")
    assert provenance.code_digest(tree) != before


def test_nothing_measured_reads_its_own_docstring():
    """The assumption the digest now rests on, asserted rather than assumed.

    Stripping docstrings is only safe while none of them is load-bearing: a
    __doc__ read at runtime would let a docstring edit change behaviour without
    changing the digest.

    Checked against the executable text, using the same stripper the digest
    uses, because a file that merely mentions __doc__ in a comment is fine and
    this file's own explanation of the rule would otherwise trip it. pytest is
    configured without --doctest-modules, so a docstring cannot execute either.
    """
    from hipbridge.verify import provenance

    for rel in provenance.CODE_PATHS:
        for path in sorted((REPO / rel).rglob("*.py")):
            if path.name in provenance.DIGEST_EXCLUDE:
                continue
            executable = provenance._executable_python(path.read_text(encoding="utf-8"))
            assert "__doc__" not in executable, f"{path} reads a docstring at runtime"


# --- the digest must fail closed -------------------------------------------


def test_a_tree_with_no_measured_code_raises_rather_than_hashing_nothing(tmp_path):
    """The bug this replaced: sha256 of the empty set is a valid-looking digest.

    `code_digest` resolved its root as `Path(__file__).parents[3]`, which is the
    repo root for `src/hipbridge/verify/provenance.py` and `lib/` for the same
    file installed in site-packages. There it matched none of CODE_PATHS, hashed
    nothing, and returned e3b0c44298fc1c14 - stamped into every report written
    from an installed copy, and compared against every committed one.
    """
    from hipbridge.verify import provenance

    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()

    with pytest.raises(provenance.DigestUnavailable) as excinfo:
        provenance.code_digest(root=tmp_path)

    assert "no measured code" in str(excinfo.value)


def test_the_empty_digest_is_never_returned(tmp_path):
    """Stated as the property rather than the mechanism, so a rewrite keeps it."""
    import hashlib

    from hipbridge.verify import provenance

    empty = hashlib.sha256(b"").hexdigest()[:16]
    with pytest.raises(provenance.DigestUnavailable):
        provenance.code_digest(root=tmp_path)
    assert provenance.code_digest(root=REPO) != empty


def test_a_partial_tree_still_digests(tmp_path):
    """Strict about nothing, not about everything.

    A tree holding only `examples/` is a legitimate thing to digest, and the
    freshness tests build exactly that. The rule is that at least one measured
    file was actually read, not that every CODE_PATHS entry exists.
    """
    from hipbridge.verify import provenance

    (tmp_path / "examples").mkdir()
    (tmp_path / "examples" / "k.cu").write_text("__global__ void k() {}", encoding="utf-8")

    assert provenance.code_digest(root=tmp_path)


def test_a_report_that_could_not_digest_is_unknown_not_fresh(tmp_path):
    """The direction to fail in: unanswerable never reads as verified."""
    from hipbridge.verify import provenance

    report = (
        "| hipbridge | 0.1 at commit `abc1234` |\n"
        f"| measured code | {provenance.DIGEST_UNAVAILABLE} (no source tree found) |\n"
    )
    verdict, why = provenance.staleness(report)

    assert verdict == "unknown"
    assert "could not be read" in why


def test_the_header_records_that_it_could_not_check(monkeypatch):
    """A report still writes from a tree we cannot digest. It just cannot claim one."""
    from hipbridge.verify import provenance

    def refuse(root=None):
        raise provenance.DigestUnavailable("no source tree found above nowhere")

    monkeypatch.setattr(provenance, "code_digest", refuse)
    head = provenance.header("t", "hipcc", "gfx942")

    assert f"| measured code | {provenance.DIGEST_UNAVAILABLE}" in head
    assert provenance.parse_code_digest(head) is None
    assert provenance.staleness(head)[0] == "unknown"

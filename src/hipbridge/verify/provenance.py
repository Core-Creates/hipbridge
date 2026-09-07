"""Where a measurement came from, recorded with the measurement.

Every number in this project's README got there because a human copied it out of
a terminal. The MI300X used for it was a rented droplet that went down twice
mid-session, taking its untracked report file with it both times. A result that
lives only in scrollback is not evidence, it is an anecdote with good intentions.

So reports carry a header naming the machine, the GPU, the toolchain and library
versions, and the commit the code was at. That last one is what makes a result
checkable later: without it, "84/84" describes an unknown program.

Every field degrades to "unknown" rather than raising. A report that fails to
write because the GPU name could not be read would be a poor trade.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from hipbridge import __version__

# Where tracked results live. Deterministic filenames, no timestamps: re-running
# a measurement should show up as a diff against the last one, not as a new file
# nobody compares. Git already keeps the history.
RESULTS_DIR = "results"


def _run(cmd: list[str]) -> str:
    if not shutil.which(cmd[0]):
        return ""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return ""
    return (r.stdout or r.stderr or "").strip()


def commit() -> str:
    """The commit this measurement was produced at, marked if the tree is dirty.

    Results themselves do not count as dirt. Writing a report necessarily
    modifies the tree, so counting it would stamp every measurement `-dirty`,
    including the ones taken from a pristine checkout, and a marker that is
    always on carries no information. What matters is whether the *code* that
    produced the number differed from the commit named beside it.
    """
    sha = _run(["git", "rev-parse", "--short", "HEAD"]).split("\n")[0]
    if not sha:
        return "unknown"
    changes = [
        line
        for line in _run(["git", "status", "--porcelain"]).splitlines()
        if line.strip() and RESULTS_DIR + "/" not in line.replace("\\", "/")
    ]
    return f"{sha}-dirty" if changes else sha


def toolchain_version(toolchain: str) -> str:
    out = _run([toolchain, "--version"])
    for line in out.splitlines():
        if "version" in line.lower():
            return line.strip()
    return out.splitlines()[0].strip() if out else "unknown"


def device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "no device visible to torch"
    except Exception:
        return "unknown"


def torch_version() -> str:
    try:
        import torch

        return torch.__version__
    except Exception:
        return "not installed"


def header(title: str, toolchain: str, arch: str) -> str:
    """A markdown provenance block for the top of a report."""
    rows = [
        ("generated", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")),
        ("hipbridge", f"{__version__} at commit `{commit()}`"),
        ("host", f"{platform.node()} ({platform.system()} {platform.machine()})"),
        ("device", device()),
        ("toolchain", f"{toolchain}, {toolchain_version(toolchain)}"),
        ("arch", arch or "default"),
        ("torch", torch_version()),
    ]
    lines = [f"# {title}", "", "| field | value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    lines.append("")
    return "\n".join(lines)


def write(path: str | Path, title: str, toolchain: str, arch: str, body: str) -> Path:
    """Write a report with its provenance header, creating the directory."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(header(title, toolchain, arch) + "\n" + body.rstrip() + "\n", encoding="utf-8")
    return p


def default_path(command: str, toolchain: str, arch: str) -> str:
    """`results/verify-hipcc-gfx942.md` and friends."""
    parts = [command, toolchain, arch or "default"]
    return str(Path(RESULTS_DIR) / ("-".join(parts) + ".md"))


# Paths whose contents a measurement is actually about. A report is stale as
# evidence when the code it measured changed, not when any commit landed: a
# README edit does not invalidate a number, and a new kernel does.
CODE_PATHS = (
    "src/hipbridge/kernels",
    "src/hipbridge/verify",
    "examples",
)


def parse_commit(report: str) -> str | None:
    """The commit a report says it was produced at, or None if it does not say."""
    m = re.search(r"\|\s*hipbridge\s*\|[^|]*commit `([0-9a-f]+)(-dirty)?`", report)
    return m.group(1) if m else None


def last_code_change() -> str | None:
    """The most recent commit touching anything a measurement depends on."""
    sha = _run(["git", "log", "-1", "--format=%h", "--", *CODE_PATHS])
    return sha.splitlines()[0] if sha.strip() else None


def known(sha: str) -> bool:
    """Is this commit present in the local history?

    CI checks out one commit by default, so an ancestry question about anything
    older is unanswerable there. Unanswerable is not the same as stale, and a
    freshness check that fails on a shallow clone would be turned off within a
    week.
    """
    return bool(_run(["git", "cat-file", "-t", sha]).startswith("commit"))


def is_ancestor(older: str, newer: str) -> bool:
    """Does `older` come at or before `newer` in history?"""
    try:
        r = subprocess.run(
            ["git", "merge-base", "--is-ancestor", older, newer],
            capture_output=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0


def staleness(report: str) -> tuple[str, str]:
    """Judge one report. Returns (verdict, explanation).

    Verdicts: "fresh", "stale", "unknown". Unknown covers a shallow clone or a
    missing git, and is deliberately distinct from stale so a caller can decide
    whether an unanswerable question should fail anything.
    """
    stamped = parse_commit(report)
    if stamped is None:
        return "unknown", "the report does not name the commit it was produced at"

    code = last_code_change()
    if code is None:
        return "unknown", "git history is unavailable here"
    if not (known(stamped) and known(code)):
        return "unknown", f"commit {stamped} is not in this clone's history"

    if is_ancestor(code, stamped):
        return "fresh", f"no measured code has changed since {stamped}"
    return (
        "stale",
        f"measured at {stamped}, but {code} has since changed something it measures",
    )


__all__ = [
    "CODE_PATHS",
    "RESULTS_DIR",
    "commit",
    "default_path",
    "device",
    "header",
    "is_ancestor",
    "known",
    "last_code_change",
    "parse_commit",
    "staleness",
    "toolchain_version",
    "torch_version",
    "write",
]

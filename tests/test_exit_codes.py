"""Exit codes are the machine-readable half of this CLI's contract.

A pipeline reads `$?`, not prose. Before this, `port` returned 0 both when it
proved a substitution and when it could not prove one, so a job on a machine
without hipcc read "unproven" as success while the text on screen said the
opposite. Its own docstring had promised 5 for that since it was written.

The distinction these pin is between a claim that failed and a claim that was
never tested. Both are "not proved" and they call for opposite responses: one is
a defect to go fix, the other a machine to go find.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hipbridge import cli, verify
from hipbridge.cli import main

needs_verify = pytest.mark.skipif(not verify.available(), reason="[verify] extra not installed")

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_the_codes_are_distinct():
    """Two outcomes sharing a code is how the original defect was possible."""
    codes = [
        cli.EXIT_OK,
        cli.EXIT_USAGE,
        cli.EXIT_REFUSED,
        cli.EXIT_NOTHING,
        cli.EXIT_REJECTED,
        cli.EXIT_UNPROVABLE,
    ]
    assert len(set(codes)) == len(codes)
    assert cli.EXIT_OK == 0, "success is 0 or nothing understands it"


def test_a_report_succeeds_whether_or_not_it_recognized_anything():
    """inspect makes no claim, so an unrecognized kernel is a result, not a failure.

    Scripts that need the distinction read the report, or `--json` when it
    lands. Overloading the exit code of an informational command is how the
    meaning of an exit code gets diluted in the first place.
    """
    assert main(["inspect", str(EXAMPLES / "row_softmax.cu")]) == cli.EXIT_OK
    assert main(["inspect", str(EXAMPLES / "tiled_transpose.cu")]) == cli.EXIT_OK


def test_bad_input_is_usage():
    assert main(["inspect", str(EXAMPLES / "does_not_exist.cu")]) == cli.EXIT_USAGE


@needs_verify
def test_port_cannot_report_success_without_a_proof():
    """The defect this file exists for.

    No hipcc on this machine, so nothing can be proved. It printed "The
    substitution is unproven, so it is not recommended" and exited 0.
    """
    rc = main(["port", str(EXAMPLES / "row_softmax.cu"), "--toolchain", "hipcc"])
    assert rc == cli.EXIT_UNPROVABLE


@needs_verify
def test_nothing_proposed_is_distinct_from_nothing_proved():
    """A kernel nobody recognized and a kernel nobody could test are not the same."""
    rc = main(["port", str(EXAMPLES / "tiled_transpose.cu"), "--toolchain", "hipcc"])
    assert rc == cli.EXIT_NOTHING


@needs_verify
def test_a_skip_stays_a_success_unless_absence_is_declared_fatal(capsys):
    """The free dry run depends on this.

    amd-verify.yml runs on a hosted runner with no GPU to prove the plumbing
    without spending anything, and every suite reporting SKIP has to leave that
    job green. --require is how a caller says a missing device is a failure.
    """
    assert main(["verify", "--toolchain", "hipcc", "--limit", "1"]) == cli.EXIT_OK
    capsys.readouterr()
    assert (
        main(["verify", "--toolchain", "hipcc", "--limit", "1", "--require"]) == cli.EXIT_UNPROVABLE
    )

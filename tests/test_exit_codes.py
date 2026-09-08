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


# --- the other half of the machine-readable surface -----------------------


def _run_json(capsys, argv):
    import json

    code = main([*argv, "--json"])
    out = capsys.readouterr().out
    return code, json.loads(out)


def test_inspect_json_needs_no_extra(capsys):
    """It has to work on a core install, so nothing in its path may reach torch.

    An earlier test of mine imported verify.suites from the torch-free half of a
    file and took the whole core matrix down with it. This is the same trap one
    layer out: `inspect` is a core command and its serializer lives in
    frontend/ir.py for that reason.
    """
    code, payload = _run_json(capsys, ["inspect", str(EXAMPLES / "row_softmax.cu")])

    assert code == cli.EXIT_OK
    assert payload["command"] == "inspect"
    assert payload["exit_code"] == cli.EXIT_OK
    kernel = payload["kernels"][0]
    assert kernel["kernel"] == "row_softmax"
    assert kernel["pattern"] == "reduce_serial"
    assert kernel["recognized"] is True
    assert kernel["facts"]["calls"], "the structural read should travel with the verdict"


def test_json_replaces_the_prose_rather_than_joining_it(capsys):
    """A caller parsing stdout should not have to skip a report first."""
    code, payload = _run_json(capsys, ["inspect", str(EXAMPLES / "tiled_transpose.cu")])

    assert code == cli.EXIT_OK
    assert payload["kernels"][0]["recognized"] is False
    assert payload["kernels"][0]["confidence"] is None


def test_the_exit_code_travels_inside_the_document(capsys):
    """So a consumer that captured stdout alone still knows what happened."""
    code, payload = _run_json(capsys, ["inspect", str(EXAMPLES / "row_softmax.cu")])
    assert payload["exit_code"] == code


@needs_verify
def test_port_json_says_unproven_rather_than_merely_omitting_proof(capsys):
    """The JSON has to be as explicit as the exit code, or it repeats the defect."""
    code, payload = _run_json(capsys, ["port", str(EXAMPLES / "layer_norm.cu")])

    assert code == cli.EXIT_UNPROVABLE
    assert payload["proved"] is False
    assert payload["unprovable"], "no reason given for an unprovable claim"
    # The proposal is still reported: what was going to be tried is useful even
    # when it could not be tried.
    assert payload["proposal"]["suite"] == "layer_norm"
    assert payload["proposal"]["epsilon"] == pytest.approx(1e-5)


@needs_verify
def test_port_json_reports_a_declined_proposal(capsys):
    code, payload = _run_json(capsys, ["port", str(EXAMPLES / "tiled_transpose.cu")])

    assert code == cli.EXIT_NOTHING
    assert payload["proposal"] is None
    assert payload["proved"] is False


@needs_verify
def test_verify_json_lists_every_suite_including_the_skipped(capsys):
    """A suite that did not run is a fact about the run, not an absence."""
    code, payload = _run_json(capsys, ["verify", "--limit", "1"])

    assert code == cli.EXIT_OK
    assert len(payload["suites"]) == 7
    assert all(s["skipped_reason"] for s in payload["suites"]), payload["suites"]


def test_the_document_is_strict_json(capsys):
    """No NaN or Infinity tokens, which json.dumps emits by default and parsers reject.

    An error of `inf` is a real outcome: it is what a candidate scores when it
    returns NaN where the oracle returns a number. It serializes as null, and
    the verdict beside it says what happened.
    """
    assert cli._json_safe(float("inf")) is None
    assert cli._json_safe(float("nan")) is None
    assert cli._json_safe({"a": [1.0, float("-inf")]}) == {"a": [1.0, None]}
    assert cli._json_safe(2.5) == 2.5


@needs_verify
def test_bench_json_carries_the_envelope_even_with_nothing_to_time(capsys):
    """A run that timed nothing still has to say what it was asked to do."""
    code, payload = _run_json(capsys, ["bench", "--shapes", "4x64"])

    assert code in (cli.EXIT_OK, cli.EXIT_REJECTED)
    assert payload["shapes"] == [[4, 64]]
    assert payload["toolchain"]
    assert isinstance(payload["measurements"], list)


@needs_verify
def test_a_bench_row_reports_the_two_facts_that_gate_quoting_it():
    """verified and comparable decide whether any of these numbers may be used.

    A shape that failed verification was timed but is not a result. A candidate
    that ran on the host is not comparable to a baseline that ran on device, and
    the first version of this command cheerfully reported 3.6x for exactly that
    mismatch. Both are fields rather than something a consumer reconstructs.
    """
    from hipbridge.verify import bench

    fast = bench.Stat(runs=(0.002, 0.002, 0.002))
    slow = bench.Stat(runs=(0.010, 0.010, 0.010))

    on_device = bench.ShapeRow(
        shape=(4, 64),
        candidate=bench.Measurement("candidate", fast, "cuda"),
        baselines=(bench.Measurement("original", slow, "cuda"),),
        verified=True,
    ).as_dict()
    assert on_device["comparable"] is True
    assert on_device["verified"] is True
    assert on_device["speedup"]["original"] == pytest.approx(5.0)
    assert on_device["candidate"]["median_us"] == pytest.approx(2.0)

    on_host = bench.ShapeRow(
        shape=(4, 64),
        candidate=bench.Measurement("candidate", fast, "cpu"),
        baselines=(bench.Measurement("original", slow, "cuda"),),
        verified=True,
    ).as_dict()
    assert on_host["comparable"] is False
    assert on_host["speedup"]["original"] is None, "a CPU-vs-GPU ratio was reported"

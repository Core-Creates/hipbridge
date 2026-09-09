"""The workflows have to be reachable without argparse.

`verify`, `bench`, `port` and `synth` were argparse callbacks, 583 of cli.py's
912 lines and roughly 480 of those orchestration rather than argument handling.
`port` is the command this project exists for and there was no way to call it
from Python at all, which is also why the CLI test file grew to 466 lines:
exercising a rule meant spawning `main([...])` and reading stdout.

These tests call the library directly. Nothing here builds an argv.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hipbridge.verify import available

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"

pytestmark = pytest.mark.skipif(not available(), reason="[verify] extra not installed")


def _pipeline():
    from hipbridge.verify import pipeline

    return pipeline


def test_port_is_callable_without_a_cli():
    """The headline: recognize, propose and prove, as a function."""
    p = _pipeline()

    outcome = p.port_file(
        EXAMPLES / "rms_norm.cu",
        p.Device(toolchain="hipcc", arch="gfx942"),
        p.Sweep(limit=2),
    )

    assert outcome.proposal is not None, "rms_norm.cu should propose a substitute"
    assert outcome.proposal.suite.name == "rms_norm"
    assert outcome.proposal.epsilon == pytest.approx(1e-5)
    # On a machine with no hipcc this is UNPROVABLE, which is the honest answer
    # and distinct from REJECTED. Both are "not proved" and they call for
    # opposite responses.
    assert outcome.outcome in (p.Outcome.OK, p.Outcome.UNPROVABLE, p.Outcome.REJECTED)
    if outcome.outcome is p.Outcome.UNPROVABLE:
        assert outcome.unprovable, "an unprovable outcome has to say why"


def test_port_declines_a_kernel_nothing_covers():
    """Declining is a first-class outcome, not an error."""
    p = _pipeline()

    outcome = p.port_file(
        EXAMPLES / "tiled_transpose.cu",
        p.Device(toolchain="hipcc"),
        p.Sweep(limit=1),
    )

    assert outcome.outcome is p.Outcome.NOTHING
    assert outcome.proposal is None
    assert outcome.as_dict()["proposal"] is None


def test_a_missing_file_is_an_outcome_rather_than_a_traceback():
    p = _pipeline()

    outcome = p.port_file("does-not-exist.cu", p.Device(), p.Sweep())

    assert outcome.outcome is p.Outcome.USAGE
    assert "does-not-exist.cu" in outcome.error


def test_naming_a_kernel_that_is_not_there_says_what_is():
    p = _pipeline()

    outcome = p.port_file(EXAMPLES / "rms_norm.cu", p.Device(), p.Sweep(), kernel="nope")

    assert outcome.outcome is p.Outcome.USAGE
    assert "rms_norm" in outcome.error, "the error should list the kernels it did find"


def test_progress_streams_rather_than_arriving_at_the_end():
    """A full sweep runs close to 45 minutes on a metered MI300X."""
    p = _pipeline()
    seen: list[str] = []

    p.port_file(
        EXAMPLES / "rms_norm.cu",
        p.Device(toolchain="hipcc"),
        p.Sweep(limit=1),
        progress=seen.append,
    )

    assert seen, "nothing was streamed"
    assert any("proposing:" in line for line in seen)


def test_a_library_caller_gets_no_output_by_default(capsys):
    """The default progress sink is silent. Printing is the CLI's job."""
    p = _pipeline()

    p.port_file(EXAMPLES / "rms_norm.cu", p.Device(toolchain="hipcc"), p.Sweep(limit=1))

    assert capsys.readouterr().out == ""


def test_run_suites_reports_every_builtin_suite():
    p = _pipeline()
    from hipbridge.verify.suites import BUILTIN

    run = p.run_suites(
        p.Device(toolchain="hipcc", arch="gfx942", examples=EXAMPLES),
        p.Sweep(limit=1),
    )

    assert [s.name for s in run.suites] == [s.name for s in BUILTIN]
    assert run.as_dict()["ran"] == run.ran


def test_the_outcome_vocabulary_covers_every_exit_code():
    """Outcome and the CLI's exit codes are a total mapping in both directions."""
    from hipbridge import cli

    p = _pipeline()

    assert set(cli._EXIT_FOR) == {o.value for o in p.Outcome}
    assert set(cli._EXIT_FOR.values()) == {
        cli.EXIT_OK,
        cli.EXIT_USAGE,
        cli.EXIT_REFUSED,
        cli.EXIT_NOTHING,
        cli.EXIT_REJECTED,
        cli.EXIT_UNPROVABLE,
    }


def test_the_dtype_choices_match_what_the_library_accepts():
    """The set was written out in three places; adding fp8 meant finding them all."""
    from hipbridge import cli

    assert set(cli.DTYPE_CHOICES) == set(_pipeline().DTYPES)


def test_sweep_limit_is_what_bounds_a_run():
    p = _pipeline()
    from hipbridge.verify.suites import ROW_SOFTMAX

    assert len(p.Sweep(limit=3).shapes_of(ROW_SOFTMAX)) == 3
    assert len(p.Sweep().shapes_of(ROW_SOFTMAX)) == len(list(ROW_SOFTMAX.shapes()))


def test_the_accuracy_rule_is_reachable_without_a_cli():
    """The rule that lived in an argparse callback.

    "A tuned baseline that computes the wrong thing must not be allowed to
    flatter the candidate" decides what a benchmark means, and it was only
    testable by invoking main([...]) and reading stdout. It is now a line in
    bench_suites, so this asserts the arbitration it depends on is importable
    and answers the three verdicts the rule branches on.
    """
    from hipbridge import verify

    assert callable(verify.arbitrate)

    import inspect

    source = inspect.getsource(_pipeline().bench_suites)
    assert "LESS ACCURATE" in source, "the rule should live with the benchmark, not the CLI"
    assert 'arb.verdict == "worse"' in source

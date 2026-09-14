"""What someone who installed the package meets on their first command.

`pip install hipbridge[verify]` then `hipbridge verify` used to raise
FileNotFoundError from inside the sweep - a bare traceback naming
examples/row_softmax.cu, from a tool whose whole argument is that it fails
closed with a report. The .cu sources are fixtures for the built-in suites and
are deliberately not packaged, so an installed copy has no examples directory
and every install would have hit this.

No torch, no device. These run on the core legs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hipbridge.cli import EXIT_USAGE, main

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("command", ["verify", "bench"])
def test_a_missing_examples_directory_is_explained_not_raised(
    command, tmp_path, capsys, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    code = main([command, "--toolchain", "hipcc"])
    err = capsys.readouterr().err

    assert code == EXIT_USAGE, "a missing fixture directory is bad input, not a crash"
    assert "no such directory" in err
    # The three things a reader needs: what is missing, why, and what to do.
    assert "not shipped in the wheel" in err, "say why it is absent, or it reads as a bug"
    assert "--examples" in err, "name the flag that fixes it"
    assert "hipbridge port" in err, (
        "point at the command that needs no examples, which is what someone with "
        "their own kernel actually wants"
    )


@pytest.mark.parametrize("command", ["verify", "bench"])
def test_an_examples_directory_without_sources_is_caught_too(command, tmp_path, capsys):
    empty = tmp_path / "examples"
    empty.mkdir()

    code = main([command, "--toolchain", "hipcc", "--examples", str(empty)])

    assert code == EXIT_USAGE
    assert "no .cu files" in capsys.readouterr().err, (
        "a directory that exists but holds nothing fails later and less clearly"
    )


def test_the_repository_checkout_still_runs():
    """The guard must not fire where the fixtures are, which is most of CI."""
    code = main(["verify", "--toolchain", "hipcc", "--examples", str(REPO / "examples")])

    # 0 with no device present: absence is a skip unless --require says otherwise.
    assert code != EXIT_USAGE


def test_port_needs_no_examples(tmp_path, monkeypatch):
    """It proves the caller's own kernel, so it never reads the fixtures.

    Guarding it too would have been the easy symmetry and the wrong one.
    """
    kernel = tmp_path / "mine.cu"
    kernel.write_text(
        "__global__ void mine(const float* in, float* out, int rows, int cols) {\n"
        "  int r = blockIdx.x; float s = 0.f;\n"
        "  for (int i = 0; i < cols; ++i) s += in[r * cols + i];\n"
        "  for (int i = 0; i < cols; ++i) out[r * cols + i] = in[r * cols + i] / s;\n"
        "}\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert main(["port", str(kernel), "--toolchain", "hipcc"]) != EXIT_USAGE

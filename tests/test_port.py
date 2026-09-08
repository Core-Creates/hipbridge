"""`port` is the command that prints SUBSTITUTION PROVED, and it had no tests.

It is the loudest claim this project makes and the only path that calls
`oracle_for` and `candidate_for` with a bound epsilon. That combination cost a
metered GPU run: a `TypeError` in `_layer_norm_affine_oracle` surfaced 17
minutes into hardware because nothing on CPU had ever executed the path.

Nothing here needs a device. `TorchReference` already satisfies the `Reference`
protocol, so a fake standing in for `NativeReference` lets the whole pipeline
run on the host: recognize, propose, probe, verify, report.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hipbridge import cli, verify
from hipbridge.cli import main

pytestmark = pytest.mark.skipif(not verify.available(), reason="[verify] extra not installed")

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _stub(monkeypatch, impl, *, available=True, reason="hipcc not on PATH"):
    """Put a host-side reference where port expects a compiled one.

    port does `from hipbridge import verify` then `verify.NativeReference(...)`,
    so patching the attribute on the module is enough; no refactor needed to
    make the command testable, which is worth knowing before anyone proposes
    one.
    """

    class Fake(verify.Reference):
        name = "fake native"

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            Fake.last = self

        def availability(self):
            return verify.Availability(available, reason)

        def __call__(self, *inputs):
            return impl(*inputs)

    monkeypatch.setattr(verify, "NativeReference", Fake)
    return Fake


def _torch_softmax(x):
    import torch

    return torch.softmax(x, dim=-1)


def _torch_rms_norm(x, eps=1e-5):
    import torch

    return x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + eps)


# --- the claim ------------------------------------------------------------


def test_a_proved_substitution_says_so_and_says_how_to_use_it(monkeypatch, capsys):
    _stub(monkeypatch, _torch_softmax)

    code = main(["port", str(EXAMPLES / "row_softmax.cu"), "--limit", "2"])
    out = capsys.readouterr().out

    assert code == cli.EXIT_OK
    assert "SUBSTITUTION PROVED" in out
    # The usage lines are per suite for a reason: port once printed a softmax
    # snippet for every kernel it proved, so a proved LayerNorm came with
    # instructions to call softmax on it.
    assert "softmax_rowwise" in out
    assert "layer_norm" not in out


def test_the_proof_reaches_json_with_the_epsilon_it_used(monkeypatch, capsys):
    _stub(monkeypatch, _torch_rms_norm)

    code = main(["port", str(EXAMPLES / "rms_norm.cu"), "--limit", "2", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert code == cli.EXIT_OK
    assert payload["proved"] is True
    assert payload["proposal"]["epsilon"] == pytest.approx(1e-5)
    assert payload["summary"]["ok"] is True
    assert payload["summary"]["cases"] > 0


# --- the refusals ---------------------------------------------------------


def test_a_garbage_reference_is_refused_with_the_launch_it_inferred(monkeypatch, capsys):
    """The failure that motivated the probe: a wrong block size read
    uninitialised shared memory and the candidate scored 3.9e75x better."""
    import torch

    _stub(monkeypatch, lambda x: torch.full_like(x, 7.0))

    code = main(["port", str(EXAMPLES / "row_softmax.cu"), "--limit", "2"])
    out = capsys.readouterr().out

    assert code == cli.EXIT_REJECTED
    assert "REFERENCE IS NOT SANE" in out
    assert "--block" in out, "the only command that inferred the launch should say so"
    assert "SUBSTITUTION PROVED" not in out


def test_a_reference_that_raises_is_diagnosed_not_traced(monkeypatch, capsys):
    def explodes(_x):
        raise RuntimeError("hipcc: error: unknown target 'gfx942'")

    _stub(monkeypatch, explodes)

    code = main(["port", str(EXAMPLES / "row_softmax.cu"), "--limit", "2"])
    out = capsys.readouterr().out

    assert code == cli.EXIT_REJECTED
    assert "could not be run" in out
    assert "gfx942" in out, "the toolchain's own words are the useful part"


def test_a_wrong_candidate_is_rejected_rather_than_reported(monkeypatch, capsys):
    """A sane reference, a candidate that does not match it: the ordinary failure."""
    import torch

    from hipbridge.verify import suites

    _stub(monkeypatch, _torch_softmax)
    monkeypatch.setattr(
        suites, "candidate_for", lambda suite, eps=None: (lambda x: torch.zeros_like(x), "broken")
    )

    code = main(["port", str(EXAMPLES / "row_softmax.cu"), "--limit", "2"])
    out = capsys.readouterr().out

    assert code == cli.EXIT_REJECTED
    assert "SUBSTITUTION REJECTED" in out
    assert "SUBSTITUTION PROVED" not in out


def test_no_device_is_unprovable_not_success(monkeypatch, capsys):
    _stub(monkeypatch, _torch_softmax, available=False)

    code = main(["port", str(EXAMPLES / "row_softmax.cu")])
    out = capsys.readouterr().out

    assert code == cli.EXIT_UNPROVABLE
    assert "CANNOT PROVE" in out


# --- the flags ------------------------------------------------------------


def test_kernel_selects_by_name_rather_than_by_position(monkeypatch, tmp_path, capsys):
    """`facts = kernels[0]` picks whatever came first, which need not be the kernel.

    The parser defines __global__ away, so a helper defined above the real
    kernel is returned first and analyzed instead of it. --kernel is the way out
    until the frontend can tell them apart.
    """
    _stub(monkeypatch, _torch_softmax)
    source = (EXAMPLES / "row_softmax.cu").read_text(encoding="utf-8")
    two = source.replace("row_softmax", "helper_first", 1) + "\n" + source
    path = tmp_path / "two_kernels.cu"
    path.write_text(two, encoding="utf-8")

    code = main(["port", str(path), "--kernel", "row_softmax", "--limit", "2", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert code == cli.EXIT_OK
    assert payload["recognition"]["kernel"] == "row_softmax"


def test_eps_declares_a_constant_the_parser_cannot_read(monkeypatch, tmp_path, capsys):
    _stub(monkeypatch, lambda x: _torch_rms_norm(x, eps=1e-6))
    source = (EXAMPLES / "rms_norm.cu").read_text(encoding="utf-8")
    hidden = source.replace(
        "float inv = rsqrtf(acc / (float)cols + 1e-5f);",
        "float e = 1e-6f;\n    float inv = rsqrtf(acc / (float)cols + e);",
    )
    path = tmp_path / "hidden_eps.cu"
    path.write_text(hidden, encoding="utf-8")

    # Without it, the substitution is declined rather than guessed at.
    assert main(["port", str(path), "--limit", "2"]) == cli.EXIT_NOTHING
    capsys.readouterr()

    code = main(["port", str(path), "--limit", "2", "--eps", "1e-6", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == cli.EXIT_OK
    assert payload["proposal"]["epsilon"] == pytest.approx(1e-6)


def test_report_writes_provenance_a_later_reader_can_check(monkeypatch, tmp_path, capsys):
    from hipbridge.verify import provenance

    _stub(monkeypatch, _torch_softmax)

    code = main(
        [
            "port",
            str(EXAMPLES / "row_softmax.cu"),
            "--limit",
            "2",
            "--report",
            str(tmp_path / "port-report.md"),
        ]
    )
    capsys.readouterr()

    assert code == cli.EXIT_OK
    written = tmp_path / "port-report.md"
    assert written.exists(), list(tmp_path.iterdir())
    text = written.read_text(encoding="utf-8")
    assert provenance.parse_code_digest(text) == provenance.code_digest()
    assert "row_softmax" in text

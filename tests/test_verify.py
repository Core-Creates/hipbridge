"""Harness tests. Skipped cleanly when the [verify] extra is absent."""

from __future__ import annotations

import pytest

from hipbridge import verify

pytestmark = pytest.mark.skipif(not verify.available(), reason="[verify] extra not installed")


def test_detects_identity_kernel():
    import torch

    from hipbridge.verify.compare import check

    x = torch.randn(64, 128)
    identity = lambda t: t.clone()  # noqa: E731  the bug we are catching
    reference = lambda t: torch.softmax(t, -1)  # noqa: E731

    r = check("identity-softmax", identity, reference, (x,))
    assert not r.passed
    assert any("IDENTITY" in f for f in r.failures)


def test_detects_dropped_max_subtraction():
    import torch

    from hipbridge.verify.compare import check

    x = torch.randn(32, 256) * 50  # large logits: unstable softmax overflows
    unstable = lambda t: torch.exp(t) / torch.exp(t).sum(-1, keepdim=True)  # noqa: E731
    reference = lambda t: torch.softmax(t, -1)  # noqa: E731

    r = check("unstable-softmax", unstable, reference, (x,))
    assert not r.passed


def test_accepts_a_correct_implementation():
    import torch

    from hipbridge.verify.compare import check

    x = torch.randn(32, 256)
    r = check("correct", lambda t: torch.softmax(t, -1), lambda t: torch.softmax(t, -1), (x,))
    assert r.passed
    assert r.max_ulp == 0


def test_ulp_diff_is_scale_free():
    import torch

    from hipbridge.verify.compare import ulp_diff

    a = torch.tensor([1.0, 1e10])
    b = torch.tensor([1.0, 1e10])
    assert int(ulp_diff(a, b).max()) == 0

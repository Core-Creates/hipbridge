"""Shape enumeration runs on core alone; input generation needs the extra.

Kept in one file deliberately, with the split expressed per-test rather than
per-module, so the torch-free half is exercised by the no-extras CI job. A
module-level skipif would hide it there.
"""

from __future__ import annotations

import pytest

from hipbridge import verify
from hipbridge.verify import shapes as sh

needs_verify = pytest.mark.skipif(not verify.available(), reason="[verify] extra not installed")


# --- torch-free: must run on a core install -------------------------------


def test_shapes_import_without_the_verify_extra():
    """Regression: shapes.py has no torch dependency and must not pretend to.

    The lazy __getattr__ in hipbridge.verify used to call require() for every
    name, so this import raised on a core install. VerifyUnavailable subclasses
    ImportError rather than AttributeError, so Python's normal submodule-import
    fallback never fired and the error escaped.
    """
    assert sh.WAVEFRONT == 64
    assert callable(sh.row_wise)


def test_row_shapes_straddle_the_wavefront():
    pairs = set(sh.row_wise())
    # 64 is the AMD wavefront. The neighbours are where mask bugs hide.
    for c in (63, 64, 65):
        assert any(col == c for _, col in pairs), f"missing column width {c}"
    assert (1, 1) in pairs, "degenerate single-element case missing"


def test_elementwise_includes_zero_and_non_multiples():
    sizes = [n for (n,) in sh.elementwise()]
    assert 0 in sizes
    assert 65 in sizes and 1_000_003 in sizes
    assert not sh.is_wavefront_aligned(65)
    assert sh.is_wavefront_aligned(1024)


def test_sample_can_truncate():
    assert len(sh.sample(sh.row_wise(), limit=5)) == 5
    assert len(sh.sample(sh.row_wise())) > 5


# --- needs torch ----------------------------------------------------------


@needs_verify
def test_generation_is_deterministic_for_a_seed():
    import torch

    a = verify.generate(verify.InputSpec((8, 8), seed=7))
    b = verify.generate(verify.InputSpec((8, 8), seed=7))
    assert torch.equal(a, b)


@needs_verify
def test_distributions_actually_differ():
    import torch

    normal = verify.generate(verify.InputSpec((64, 64), distribution=verify.Distribution.NORMAL))
    large = verify.generate(verify.InputSpec((64, 64), distribution=verify.Distribution.LARGE))
    assert large.abs().max() > normal.abs().max() * 10

    const = verify.generate(verify.InputSpec((16,), distribution=verify.Distribution.CONSTANT))
    assert torch.equal(const, torch.full((16,), 3.5))

    sparse = verify.generate(verify.InputSpec((64, 64), distribution=verify.Distribution.SPARSE))
    assert (sparse == 0).float().mean() > 0.5


@needs_verify
def test_large_distribution_overflows_naive_exp():
    """The whole reason LARGE exists: naive exp() must actually blow up on it."""
    import torch

    x = verify.generate(verify.InputSpec((4, 64), distribution=verify.Distribution.LARGE))
    assert torch.isinf(torch.exp(x)).any(), "LARGE is not large enough to catch overflow"


@needs_verify
def test_native_reference_reports_unavailable_without_a_toolchain():
    """Must degrade with a reason, never raise at construction."""
    ref = verify.NativeReference(
        source="__global__ void k(const float* a, float* b) { b[0] = a[0]; }",
        launch=verify.LaunchSpec(kernel="k", grid=lambda s: (1, 1, 1)),
        toolchain="nvcc",
    )
    a = ref.availability()
    assert isinstance(a.ok, bool)
    if not a:
        assert a.reason

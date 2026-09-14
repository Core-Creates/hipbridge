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


@pytest.mark.parametrize("width", [32, 64])
def test_both_wavefront_widths_are_straddled_on_both_axes(width):
    """RDNA is 32 wide, and the sweep only ever straddled 64.

    No row count was 31, 32 or 33 and no width was 33, so the first 32-wide
    boundary went untested. Asserted for both widths on both axes, so dropping
    either family's neighbours is caught here.
    """
    neighbours = (width - 1, width, width + 1)
    assert set(neighbours) <= set(sh._ROWS), f"row counts miss {neighbours}"
    assert set(neighbours) <= set(sh._COLS), f"column widths miss {neighbours}"


def test_a_default_run_reaches_the_rdna_boundary():
    """The CLI's default --limit is 12, so the RDNA shapes must sit inside it."""
    default = sh.sample(sh.row_wise(), limit=12)
    assert any(r == 32 and c == 33 for r, c in default), default
    assert any(r == 33 and c == 32 for r, c in default), default
    assert any(r == 31 and c == 31 for r, c in default), default


def test_the_metered_prefixes_did_not_move():
    """The MI300X record is taken at --limit 4 and the nightly at --limit 2.

    Adding shapes must not quietly change what those runs measure, or their
    committed numbers stop describing the sweep that produces them.
    """
    assert sh.sample(sh.row_wise(), limit=4) == [(1, 1), (2, 32768), (64, 65), (1000, 128)]


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


def test_a_truncated_sweep_still_spans_row_counts():
    """Regression: the sweep was rows-outer, so every prefix had exactly one row.

    A kernel that ignores its row stride, or is launched with the wrong grid, is
    correct on a single row and wrong on every other one. The metered hardware
    runs truncate at --limit 4, so that prefix is the one that has to carry real
    row coverage, and a wide row, and still be cheap enough to be worth running.
    """
    first4 = sh.sample(sh.row_wise(), limit=4)
    assert len({r for r, _ in first4}) == 4, first4
    assert max(c for _, c in first4) >= 1024, first4
    assert max(r * c for r, c in first4) <= 200_000, first4

    # The whole first pass covers every row count, and it has to fit inside the
    # CLI's default --limit of 12 for a default run to reach all of it.
    assert len(sh._FIRST_PASS) <= 12, "a default run no longer reaches the whole first pass"
    first = sh.sample(sh.row_wise(), limit=len(sh._FIRST_PASS))
    assert {r for r, _ in first} == set(sh._ROWS), first


def test_reordering_did_not_drop_any_shape():
    """The order changed; the sweep did not. Anything else would be a quiet loss."""
    pairs = list(sh.row_wise())
    expected = {(r, c) for r in sh._ROWS for c in sh._COLS if r * c <= sh._CAP}
    assert set(pairs) == expected
    assert len(pairs) == len(set(pairs)), "row_wise yielded a duplicate"


def test_the_first_pass_is_drawn_from_the_declared_axes():
    """A hand-picked prefix can drift from the axes it claims to sample."""
    for r, c in sh._FIRST_PASS:
        assert r in sh._ROWS, r
        assert c in sh._COLS, c


def test_the_sweep_reaches_past_what_one_block_can_hold():
    """A vocabulary softmax is 32k to 128k columns and the sweep stopped at 4096.

    The kernels could not run those shapes and the sweep would never have asked,
    so the gap was invisible from both sides at once.
    """
    widths = {c for _, c in sh.row_wise()}
    assert max(widths) >= 32768, sorted(widths)

    first4 = sh.sample(sh.row_wise(), limit=4)
    assert max(c for _, c in first4) > 8192, (
        "a truncated run never crosses the tiling threshold: " + str(first4)
    )

"""Static checks on the Triton kernels, which nothing else executes.

Triton has no Windows build and hosted CI has no GPU, so src/hipbridge/kernels/
runs only on the MI300X, only when someone dispatches a workflow. Between those
runs the cheapest possible defect, a launch passing the wrong number of
arguments, would sit undetected until it cost twenty minutes of metered time.

These parse the modules with `ast` rather than importing them, so they run on a
core install with neither torch nor Triton present. They cannot tell you a
kernel is correct. They can tell you it is not obviously broken, which is the
part that was free and missing.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

KERNELS = Path(__file__).resolve().parents[1] / "src" / "hipbridge" / "kernels"

# _launch dispatches through a variable, so no static reader can pair it with a
# kernel. The pairing is written down here instead of going unchecked.
INDIRECT = {"_launch": ("_layer_norm_kernel", "_rms_norm_kernel")}


def _module(name: str) -> ast.Module:
    return ast.parse((KERNELS / name).read_text(encoding="utf-8"))


def _modules():
    return sorted(p.name for p in KERNELS.glob("*.py") if p.name != "__init__.py")


def _jit_kernels(tree: ast.Module) -> dict[str, int]:
    """Each @triton.jit kernel, and how many runtime arguments it takes.

    constexpr parameters are excluded: Triton takes those as keywords.
    """
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if not any("jit" in ast.unparse(d) for d in node.decorator_list):
            continue
        constexpr = sum(
            1 for a in node.args.args if a.annotation and "constexpr" in ast.unparse(a.annotation)
        )
        out[node.name] = len(node.args.args) - constexpr
    return out


def _launches(tree: ast.Module):
    """Every `kernel[grid](...)` call, as (callee, positional count, line)."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript):
            found.append((ast.unparse(node.func.value), len(node.args), node.lineno))
    return found


@pytest.mark.parametrize("name", _modules())
def test_every_launch_passes_exactly_what_its_kernel_takes(name):
    tree = _module(name)
    kernels = _jit_kernels(tree)
    launches = _launches(tree)
    assert launches, f"{name} defines kernels nothing launches"

    for callee, count, line in launches:
        if callee in kernels:
            assert count == kernels[callee], (
                f"{name}:{line} launches {callee} with {count} arguments, "
                f"but it takes {kernels[callee]}"
            )
        else:
            # Dispatched through a variable; check against the declared pairing.
            holder = _enclosing(tree, line)
            paired = INDIRECT.get(holder, ())
            assert paired, f"{name}:{line} launches {callee} inside {holder}, which nothing pairs"
            arities = {kernels[k] for k in paired}
            assert arities == {count}, (
                f"{name}:{line} passes {count} arguments to {sorted(paired)}, "
                f"which take {sorted(arities)}"
            )


def _enclosing(tree: ast.Module, line: int) -> str:
    """The innermost function containing this line."""
    best, best_start = "", -1
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.lineno <= line <= (node.end_lineno or line) and node.lineno > best_start:
            best, best_start = node.name, node.lineno
    return best


@pytest.mark.parametrize("name", _modules())
def test_no_kernel_indexes_a_row_without_its_element_stride(name):
    """`row * row_stride + col` assumes stride(-1) == 1, silently.

    That assumption held for every tensor the sweep could build, because
    inputs.py only ever made fresh contiguous ones, so a transposed or sliced
    input returned wrong numbers with no error and nothing could catch it.
    """
    source = (KERNELS / name).read_text(encoding="utf-8")
    for bad in (
        "in_row_stride + cols,",
        "in_row_stride + 2 * i,",
        "out_row_stride + cols,",
        "out_row_stride + 2 * i,",
    ):
        assert bad not in source, f"{name} indexes without an element stride: {bad}"


def test_the_rotation_tables_do_not_share_a_stride():
    """cos_tab.stride(0) was passed for both, so sin was read with cos's stride.

    Two tables built by different paths, one a slice of a cache and one freshly
    allocated, then read correct cosines and garbage sines: a plausible
    non-rotation rather than an error.
    """
    for name in ("rope.py", "fused.py"):
        source = (KERNELS / name).read_text(encoding="utf-8")
        assert "tab_row_stride" not in source, f"{name} still shares one table stride"
        assert "sin_tab.stride(0)" in source, f"{name} never reads sin's own stride"
        assert "sin_tab.stride(1)" in source, f"{name} never reads sin's own element stride"


TILED = {"softmax.py", "norm.py"}
ROTATION = {"rope.py", "fused.py"}


def test_the_reducing_kernels_hand_wide_rows_to_the_tiled_path():
    """`BLOCK = next_power_of_2(n_cols)` has a ceiling, and rows do not.

    num_warps saturates at 16, so at 65536 columns each lane holds 64 elements
    plus temporaries and Triton either spills or fails to compile. A vocabulary
    softmax is 32k to 128k columns, which made the most common wide-row kernel
    in inference the one shape this package could not run.
    """
    for name in sorted(TILED):
        source = (KERNELS / name).read_text(encoding="utf-8")
        assert "wide.TILED_ABOVE" in source, f"{name} never checks the width ceiling"
        assert "next_power_of_2" in source, f"{name} lost its single-tile fast path"


def test_the_rotations_refuse_a_wide_row_rather_than_spilling():
    """RoPE operates on a head dimension, tens to hundreds of channels.

    A row wider than a block is a shape mistake there far more often than it is
    a rotation, so it is refused with a sentence rather than tiled or spilled.
    """
    for name in sorted(ROTATION):
        source = (KERNELS / name).read_text(encoding="utf-8")
        assert "raise ValueError" in source, f"{name} does not refuse anything"
        assert "wide.TILED_ABOVE" in source, f"{name} never checks the width"


def test_an_empty_row_is_returned_rather_than_reduced():
    """next_power_of_2(0) is 1, the mask is all false, and a reduction over an
    all -inf vector propagates NaN."""
    for name in sorted(TILED):
        source = (KERNELS / name).read_text(encoding="utf-8")
        assert "x.numel() == 0" in source, f"{name} still launches on an empty tensor"


def test_the_tiled_module_tiles_below_the_threshold_it_takes_over_at():
    """A tile larger than the ceiling would mean the tiled path never loops."""
    source = (KERNELS / "wide.py").read_text(encoding="utf-8")
    ns = {}
    for line in source.splitlines():
        if line.startswith(("TILE =", "TILED_ABOVE =")):
            exec(line, ns)  # noqa: S102 - two integer literals from our own source
    assert ns["TILE"] < ns["TILED_ABOVE"], ns
    assert ns["TILED_ABOVE"] >= 4096, "taking over below a shape with a measurement history"

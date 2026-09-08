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

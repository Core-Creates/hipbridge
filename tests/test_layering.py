"""The test that makes the future split possible.

Core (frontend, recognize, analysis, cli) must never import an extra at module
level. Violate this and drover.verify / drover.kernels can no longer be lifted
into their own distributions without a refactor, and core stops installing
without torch.

Lazy imports inside function bodies are permitted and are the intended
mechanism for optional behaviour.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "drover"
EXTRAS = {"drover.verify", "drover.kernels"}
CORE = ["frontend", "recognize", "analysis", "cli.py", "__init__.py"]


def _core_files() -> list[Path]:
    out: list[Path] = []
    for entry in CORE:
        p = SRC / entry
        out.extend(sorted(p.rglob("*.py")) if p.is_dir() else [p])
    return out


def _module_level_imports(tree: ast.Module) -> list[tuple[str, int]]:
    """Only top-level imports. Function-scoped imports are deliberately ignored."""
    found = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            found += [(a.name, node.lineno) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append((node.module, node.lineno))
    return found


@pytest.mark.parametrize("path", _core_files(), ids=lambda p: p.name)
def test_core_does_not_import_extras(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for name, lineno in _module_level_imports(tree):
        for extra in EXTRAS:
            assert not (name == extra or name.startswith(extra + ".")), (
                f"{path.relative_to(SRC)}:{lineno} imports {name} at module level. "
                f"Move it inside the function that needs it."
            )


def test_core_imports_without_extras():
    """Core must be importable with neither extra present."""
    import drover

    assert drover.__version__


def test_extras_report_availability_without_raising():
    from drover import kernels, verify

    assert isinstance(kernels.available(), bool)
    assert isinstance(verify.available(), bool)

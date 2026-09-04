"""The test that makes the future split possible.

Core (frontend, recognize, analysis, cli) must never import an extra at module
level. Violate this and hipbridge.verify / hipbridge.kernels can no longer be lifted
into their own distributions without a refactor, and core stops installing
without torch.

Lazy imports inside function bodies are permitted and are the intended
mechanism for optional behaviour.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "hipbridge"
EXTRAS = {"hipbridge.verify", "hipbridge.kernels"}
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
    import hipbridge

    assert hipbridge.__version__


def test_extras_report_availability_without_raising():
    from hipbridge import kernels, verify

    assert isinstance(kernels.available(), bool)
    assert isinstance(verify.available(), bool)


# --- extras must declare exactly what they import --------------------------

TOP_LEVEL = {"torch", "triton", "numpy", "clang", "yaml"}


def _third_party_imports(pkg_dir: Path) -> set[str]:
    """Top-level third-party modules imported anywhere under pkg_dir."""
    found: set[str] = set()
    for path in pkg_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                found.add(node.module.split(".")[0])
    return found & TOP_LEVEL


def _extra(name: str) -> set[str]:
    try:
        import tomllib  # stdlib from 3.11
    except ModuleNotFoundError:  # pragma: no cover - 3.10 only
        import tomli as tomllib

    data = tomllib.loads((SRC.parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    deps = data["project"]["optional-dependencies"][name]
    return {d.split(";")[0].split(">=")[0].split("==")[0].strip() for d in deps}


def test_verify_extra_does_not_declare_triton():
    """Regression, and it would have broken the AMD box specifically.

    verify/ imports torch and nothing else. Declaring triton there makes
    `pip install .[verify]` pull the NVIDIA-flavoured triton wheel over the
    pytorch-triton-rocm that ROCm PyTorch depends on, breaking the very GPU the
    harness was installed to test.
    """
    assert "triton" not in _extra("verify")
    assert "triton" not in _third_party_imports(SRC / "verify")


def test_each_extra_declares_what_its_package_imports():
    for extra, pkg in (("verify", "verify"), ("kernels", "kernels")):
        imported = _third_party_imports(SRC / pkg)
        declared = _extra(extra)
        # torch is supplied by the platform on ROCm images, so kernels may
        # import it without declaring it; triton may not be over-declared.
        undeclared = imported - declared - {"torch"}
        assert not undeclared, f"[{extra}] imports {undeclared} without declaring it"

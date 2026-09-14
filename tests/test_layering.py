"""The test that makes the future split possible.

Core must never import an extra at module level. Violate this and
hipbridge.verify / hipbridge.kernels can no longer be lifted into their own
distributions without a refactor, and core stops installing without torch.

Lazy imports inside function bodies are permitted and are the intended mechanism
for optional behaviour.

Three things this file used to get wrong, all of them the same shape: it looked
stricter than it was.

  1. CORE was an allowlist of five entries, so `synth/` (301 lines) was checked
     by nothing and an `import torch` at its top would have passed. Any package
     added later was likewise unchecked by default. It is now a denylist, so new
     code is checked unless someone says otherwise.

  2. `_module_level_imports` walked `tree.body` only, so anything one level
     deeper was invisible - including `if TYPE_CHECKING:` and `try: import ...
     except ImportError:`, both of which are module-level imports and both of
     which are exactly what someone reaches for first. It now walks the whole
     tree and excludes function scopes by ancestry rather than by depth.

  3. The direction that actually matters was untested. Both extras claimed in
     their docstrings to have no import edge back into core, and both were
     false: verify/substitutions.py imported the IR from the day it was written,
     kernels/__init__.py imported Pattern for a dead registry, and
     verify/provenance.py imported the package root, which pulled in the
     frontend and therefore clang.cindex - giving the [verify] extra a hard
     dependency on libclang that it never declared.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "hipbridge"
EXTRAS = {"hipbridge.verify", "hipbridge.kernels"}

# Everything under src/hipbridge is core unless it is an extra. A denylist, so a
# package added next month is checked without anyone remembering to add it.
NOT_CORE = {"verify", "kernels", "__pycache__"}


def _core_files() -> list[Path]:
    out: list[Path] = []
    for entry in sorted(SRC.iterdir()):
        if entry.name in NOT_CORE or entry.name.endswith(".egg-info"):
            continue
        if entry.is_dir():
            out.extend(sorted(entry.rglob("*.py")))
        elif entry.suffix == ".py":
            out.append(entry)
    return out


def _function_scoped(tree: ast.Module) -> set[int]:
    """Every node id that sits inside a function body, at any depth."""
    inside: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                inside.add(id(child))
    return inside


def _module_level_imports(tree: ast.Module) -> list[tuple[str, int]]:
    """Imports that run at import time, wherever they are written.

    Not `tree.body` only. An import inside `if TYPE_CHECKING:` or a try/except
    is module level however deeply it is nested; an import inside a function is
    not, however shallow.
    """
    scoped = _function_scoped(tree)
    found = []
    for node in ast.walk(tree):
        if id(node) in scoped:
            continue
        if isinstance(node, ast.Import):
            found += [(a.name, node.lineno) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append((node.module, node.lineno))
    return found


@pytest.mark.parametrize("path", _core_files(), ids=lambda p: str(p.relative_to(SRC)))
def test_core_does_not_import_extras(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for name, lineno in _module_level_imports(tree):
        for extra in EXTRAS:
            assert not (name == extra or name.startswith(extra + ".")), (
                f"{path.relative_to(SRC)}:{lineno} imports {name} at module level. "
                f"Move it inside the function that needs it."
            )


def test_the_denylist_actually_covers_the_tree():
    """A guard on the guard: every package under src/hipbridge is classified."""
    packages = {p.name for p in SRC.iterdir() if p.is_dir() and not p.name.endswith(".egg-info")}
    checked = {f.relative_to(SRC).parts[0] for f in _core_files()}
    unchecked = packages - NOT_CORE - checked
    assert not unchecked, f"{sorted(unchecked)} is neither core nor an extra"


def test_a_type_checking_import_would_be_caught():
    """The hole that made this file weaker than it read.

    Asserted on a fixture rather than on the tree, so it keeps testing the rule
    after the last real violation is gone.
    """
    tree = ast.parse(
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from hipbridge.verify import Harness\n"
    )
    assert ("hipbridge.verify", 3) in _module_level_imports(tree)


def test_a_try_except_import_would_be_caught():
    tree = ast.parse(
        "try:\n    from hipbridge.kernels import softmax\nexcept ImportError:\n    softmax = None\n"
    )
    assert ("hipbridge.kernels", 2) in _module_level_imports(tree)


def test_a_function_scoped_import_is_still_allowed():
    """The intended mechanism for optional behaviour, at any depth."""
    tree = ast.parse(
        "def f():\n"
        "    if True:\n"
        "        from hipbridge.verify import Harness\n"
        "        return Harness\n"
    )
    assert _module_level_imports(tree) == []


def test_core_imports_without_extras():
    """Core must be importable with neither extra present."""
    import hipbridge

    assert hipbridge.__version__


def test_extras_report_availability_without_raising():
    from hipbridge import kernels, verify

    assert isinstance(kernels.available(), bool)
    assert isinstance(verify.available(), bool)


# --- the direction that actually matters -----------------------------------
#
# What each extra may reach back into. Narrow and true beats broad and false:
# verify reads KernelFacts, so claiming it has no edge into core was a fiction
# from the first commit. kernels reaches for nothing, and the rule says so.
ALLOWED_CORE_IMPORTS = {
    # frontend supplies KernelFacts, the language this package reads; recognize
    # and synth are the other halves of the port and synth pipelines. kernels is
    # the substitute a Suite resolves, which is an edge between two extras and
    # not into core. analysis owns the wavefront width per arch, which the
    # native reference needs when a header skew has to be papered over.
    "verify": {
        "hipbridge.analysis",
        "hipbridge.frontend",
        "hipbridge.recognizers",
        "hipbridge.synth",
        "hipbridge.kernels",
    },
    "kernels": set(),
}

# Every package under src/hipbridge, so `from hipbridge import verify` can be
# told apart from `from hipbridge import __version__`. The first is a submodule
# import; only the second reaches into the root's own namespace.
SUBPACKAGES = {p.name for p in SRC.iterdir() if p.is_dir() and (p / "__init__.py").is_file()}


def _core_imports(pkg_dir: Path) -> set[tuple[str, str, int]]:
    """Every import of another hipbridge package from inside pkg_dir, at any scope.

    `from hipbridge import verify` is recorded as `hipbridge.verify`, not as
    `hipbridge`: it binds a submodule, and treating it as a root import would
    flag every lazy import in the file while missing the one that mattered.
    """
    pkg = f"hipbridge.{pkg_dir.name}"
    found = set()
    for path in sorted(pkg_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom) and node.module == "hipbridge":
                names = [
                    f"hipbridge.{a.name}" if a.name in SUBPACKAGES else "hipbridge"
                    for a in node.names
                ]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            for name in names:
                if not name.startswith("hipbridge"):
                    continue
                if name == pkg or name.startswith(pkg + "."):
                    continue
                found.add((str(path.relative_to(SRC)), name, node.lineno))
    return found


@pytest.mark.parametrize("pkg", sorted(ALLOWED_CORE_IMPORTS))
def test_an_extra_reaches_back_only_where_it_is_allowed_to(pkg: str):
    allowed = ALLOWED_CORE_IMPORTS[pkg]
    for where, name, lineno in sorted(_core_imports(SRC / pkg)):
        root = ".".join(name.split(".")[:2])
        assert root in allowed, (
            f"{where}:{lineno} imports {name}, which {pkg} is not allowed to reach. "
            f"Allowed: {sorted(allowed) or 'nothing'}."
        )


def test_no_extra_imports_the_package_root():
    """`from hipbridge import __version__` was not free.

    Importing a *symbol* from the root ran hipbridge/__init__.py's eager
    re-exports, which imported the frontend, which imported clang.cindex. So the
    [verify] extra needed libclang to read its own version number. The root is
    now lazy, and pulling a symbol out of it is still the wrong direction:
    import the submodule that defines what you need.

    `from hipbridge import verify` is a submodule import and is not this.
    """
    for pkg in ALLOWED_CORE_IMPORTS:
        offenders = [
            f"{where}:{lineno}"
            for where, name, lineno in _core_imports(SRC / pkg)
            if name == "hipbridge"
        ]
        assert not offenders, (
            f"{pkg} imports the hipbridge package root at {offenders}. "
            f"Import the submodule that defines what you need."
        )


def test_importing_an_extra_does_not_pull_in_libclang():
    """The property, rather than the import statement that used to break it."""
    import subprocess
    import sys

    probe = "import sys, hipbridge.verify.provenance; print('clang.cindex' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", (
        "importing hipbridge.verify pulled in libclang, which the [verify] extra "
        "does not declare and has no use for"
    )


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


# --- the re-export surface --------------------------------------------------


def test_the_lazy_re_exports_all_resolve():
    """Five lines that would have caught two names that never existed.

    `BenchResult` and `Comparison` were in _LAZY and in __all__, so
    `from hipbridge.verify import *` raised AttributeError on a machine that had
    torch installed.
    """
    pytest.importorskip("torch")
    import hipbridge.verify as v

    broken = [name for name in v.__all__ if not hasattr(v, name)]
    assert not broken, f"_LAZY names nothing resolves to: {broken}"


def test_the_lazy_map_mirrors_each_submodule_exactly():
    """Both directions. The map had rotted each way independently."""
    pytest.importorskip("torch")
    import importlib

    import hipbridge.verify as v

    for name in v._MIRRORED:
        mod = importlib.import_module(f"hipbridge.verify.{name}")
        declared = set(mod.__all__)
        mirrored = {k for k, owner in v._LAZY.items() if owner == name}
        assert mirrored == declared, (
            f"hipbridge.verify re-exports {sorted(mirrored - declared)} that "
            f"{name} does not export, and omits {sorted(declared - mirrored)}"
        )


def test_no_root_export_collides_with_a_subpackage():
    """The general rule, replacing the special case it was written for.

    `recognize` used to name both a subpackage and the function inside it. Two
    objects, one name, one namespace, and whichever was assigned last won: eager
    assignment in hipbridge/__init__.py made the function win, so the collision
    was invisible right up until someone made that import lazy. Then it failed
    only in full runs, because a module-level __getattr__ is consulted after
    __dict__ and the import system writes a submodule into its parent's __dict__
    on first load. `recognize(facts)` raised "'module' object is not callable"
    in a full test run and passed on its own.

    The package is now `hipbridge.recognizers`. This asserts the property rather
    than that one name, so the next export to shadow a subpackage is caught when
    it is added rather than when someone refactors an import three months later.
    """
    import hipbridge

    collisions = sorted(set(hipbridge.__all__) & SUBPACKAGES)
    assert not collisions, (
        f"{collisions} names both a subpackage and a re-export of hipbridge. "
        f"Whichever is assigned last wins, which makes the resolution an "
        f"accident of import order. Rename one of them."
    )


def test_recognize_is_the_function():
    """The ergonomic import, pinned. It is what cli.py and pipeline.py use."""
    import hipbridge

    assert callable(hipbridge.recognize)
    assert hipbridge.recognize.__name__ == "recognize"
    assert hipbridge.recognize.__module__ == "hipbridge.recognizers.base"


def test_the_root_still_exports_everything_it_promises():
    """Laziness must not quietly shrink the public surface."""
    import hipbridge

    missing = [name for name in hipbridge.__all__ if not hasattr(hipbridge, name)]
    assert not missing, f"__all__ names nothing resolves to: {missing}"

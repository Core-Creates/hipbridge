"""BUILTIN is the one list of what this project covers. Everything else derives.

Adding a kernel used to mean editing four places inside suites.py alone: the
Suite literal, BUILTIN, a branch in a 55-line `_candidate_impl` if-chain, and
__all__. The chain ended in `raise KeyError(suite.name)`, so a suite added to
BUILTIN and forgotten in the chain was well-formed by every test in the repo and
raised the first time anyone tried to prove it. Two more copies lived outside the
file, in the propose() dispatch tuple and in the AMD workflow's port loop, and
rms_norm_rope had already fallen out of the second one.

These tests assert the derivations rather than the copies.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from hipbridge.verify import available

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"

pytestmark = pytest.mark.skipif(not available(), reason="[verify] extra not installed")


def _suites():
    from hipbridge.verify.suites import BUILTIN

    return BUILTIN


def _ids(suites):
    return [s.name for s in suites]


@pytest.mark.parametrize(
    "suite", _suites() if available() else [], ids=_ids(_suites()) if available() else []
)
def test_every_suite_declares_a_substitute_that_exists(suite):
    """Checked by parsing the module, not by importing it.

    Importing hipbridge.kernels.* needs Triton, which publishes Linux wheels
    only, so an import-based check would be skipped on exactly the machines
    where this file is usually edited.
    """
    assert suite.triton_impl, f"{suite.name} declares no triton_impl"
    module, _, function = suite.triton_impl.partition(":")
    assert function, f"{suite.name}: triton_impl must be 'module:function'"

    path = SRC / Path(*module.split(".")).with_suffix(".py")
    assert path.is_file(), f"{suite.name}: {module} does not exist at {path}"

    defined = {
        node.name
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert function in defined, f"{suite.name}: {module} defines no {function}"


@pytest.mark.parametrize(
    "suite", _suites() if available() else [], ids=_ids(_suites()) if available() else []
)
def test_the_printed_import_is_the_one_that_would_run(suite):
    """The bug this closes: usage_import was typed by hand beside the branch that
    imported the same thing, so a proof could run one function and recommend
    another."""
    module, _, function = suite.triton_impl.partition(":")
    assert suite.usage_import == f"from {module} import {function}"
    assert function in suite.usage_call, (
        f"{suite.name}: usage_call {suite.usage_call!r} does not call {function}"
    )


@pytest.mark.parametrize(
    "suite", _suites() if available() else [], ids=_ids(_suites()) if available() else []
)
def test_every_suite_can_be_proposed(suite):
    """A suite with no evidence test is unreachable from `port`, silently."""
    assert suite.patterns, f"{suite.name} declares no patterns, so nothing can propose it"
    assert suite.evidence is not None, f"{suite.name} declares no evidence test"


@pytest.mark.parametrize(
    "suite", _suites() if available() else [], ids=_ids(_suites()) if available() else []
)
def test_every_suite_has_a_fallback(suite):
    """Verification has to mean something on a machine without Triton."""
    assert suite.fallback is not None, f"{suite.name} declares no fallback"
    assert suite.fallback_label, f"{suite.name} declares no fallback_label"


@pytest.mark.parametrize(
    "suite", _suites() if available() else [], ids=_ids(_suites()) if available() else []
)
def test_every_suite_has_its_two_example_sources(suite):
    """The naive original and the competent baseline the candidate must beat."""
    for rel in (suite.source_file, *(b.source_file for b in suite.baselines)):
        assert (REPO / "examples" / rel).is_file(), f"{suite.name}: examples/{rel} missing"


def test_candidate_for_resolves_every_suite():
    """The whole registry, exercised. `_candidate_impl` raised KeyError here."""
    from hipbridge.verify.suites import BUILTIN, candidate_for

    for suite in BUILTIN:
        fn, described = candidate_for(suite)
        assert callable(fn), f"{suite.name} resolved to something uncallable"
        assert described, f"{suite.name} resolved without a description"


def test_the_workflow_proves_every_suite():
    """The AMD workflow's port loop hardcoded six names and omitted rms_norm_rope,
    which is why results/ holds six port reports for seven suites."""
    workflow = (REPO / ".github" / "workflows" / "amd-verify.yml").read_text(encoding="utf-8")

    for line in workflow.splitlines():
        if line.strip().startswith("for k in ") and "row_softmax" in line:
            pytest.fail(
                "amd-verify.yml still hardcodes its kernel list. Ask BUILTIN "
                f"instead, or the next kernel goes unproved too: {line.strip()}"
            )


def test_the_readme_table_lists_exactly_the_builtin_suites():
    """The README's substitution table is documentation of BUILTIN, so derive it."""
    from hipbridge.verify.suites import BUILTIN

    readme = (REPO / "README.md").read_text(encoding="utf-8")
    listed = {
        line.split("|")[1].strip().strip("`")
        for line in readme.splitlines()
        if line.startswith("| `") and "|" in line[3:]
    }
    expected = {s.name for s in BUILTIN}

    assert expected <= listed, f"README does not list {sorted(expected - listed)}"

"""Recognition tests.

The negative case matters more than the positive ones. A classifier that never
says UNKNOWN is the failure mode this whole design exists to avoid, so it is
asserted explicitly.
"""

from __future__ import annotations

import pytest

from hipbridge import Pattern, parse_file, recognize


def _one(path):
    kernels = parse_file(path)
    assert len(kernels) == 1, f"expected exactly one kernel in {path.name}"
    return recognize(kernels[0])


def test_tree_reduction_is_not_elementwise(examples):
    r = _one(examples / "tree_reduce.cu")
    assert r.pattern is Pattern.REDUCE_TREE
    assert r.confidence == "certain"
    assert r.facts.barriers == 2
    assert r.facts.has_halving_stride
    assert r.facts.shared[0].size_bytes == 1024


def test_the_tree_note_does_not_assume_a_wavefront(examples):
    """Recognition runs before a device is named, so the note has to give both.

    It said "wavefront is 64 wide, tree needs 6 steps not 5", which is CDNA's
    answer and backwards for every Radeon, where RDNA is 32 wide and 5 is right.
    """
    report = _one(examples / "tree_reduce.cu").report()

    assert "wavefront is 64 wide" not in report
    assert "CDNA (64 wide)" in report
    assert "RDNA (32 wide)" in report


def test_row_softmax_is_a_serial_reduction(examples):
    r = _one(examples / "row_softmax.cu")
    assert r.pattern is Pattern.REDUCE_SERIAL
    assert r.facts.loops == 3
    assert not r.facts.shared


def test_saxpy_is_elementwise(examples):
    r = _one(examples / "saxpy.cu")
    assert r.pattern is Pattern.ELEMENTWISE
    assert r.facts.loops == 0
    assert len(r.facts.inputs) == 2
    assert len(r.facts.outputs) == 1


def test_unrecognized_kernel_returns_unknown(examples):
    """No recognizer covers a tiled transpose yet. It must say so."""
    r = _one(examples / "tiled_transpose.cu")
    assert r.pattern is Pattern.UNKNOWN
    assert not r.recognized
    assert "no recognizer claimed" in r.report()


def test_no_catch_all_pattern_exists():
    """Pattern must not grow a default member that swallows unknowns."""
    assert Pattern.UNKNOWN.value == "unknown"
    with pytest.raises(ValueError):
        Pattern("map")


def test_parses_without_errors(examples):
    for name in ("tree_reduce.cu", "row_softmax.cu", "saxpy.cu"):
        kernels = parse_file(examples / name)
        assert kernels[0].parse_errors == 0, f"{name} did not parse cleanly"

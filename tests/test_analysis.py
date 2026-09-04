from __future__ import annotations

import pytest

from drover import ARCHS, lds_padding, occupancy, roofline


def test_all_archs_are_64_wide_except_rdna():
    assert ARCHS["cdna3"].wavefront == 64
    assert ARCHS["rdna3"].wavefront == 32


def test_occupancy_is_vgpr_limited():
    full = occupancy(48, "cdna3")
    assert full["wavefronts_per_simd"] == 8
    assert full["occupancy_pct"] == 100.0

    spilled = occupancy(256, "cdna3")
    assert spilled["wavefronts_per_simd"] == 2
    assert spilled["limited_by"] == "vgpr"


def test_roofline_classifies_bound():
    mem = roofline(flops=1e9, bytes_moved=1e9, arch="cdna3")
    assert mem["bound"] == "memory"

    comp = roofline(flops=1e15, bytes_moved=1e9, arch="cdna3")
    assert comp["bound"] == "compute"
    assert comp["attainable_tflops"] <= comp["peak_tflops"]


def test_analysis_labels_itself_as_datasheet_not_measurement():
    """Guards against these numbers ever being presented as benchmark results."""
    assert "not measured" in occupancy(48)["basis"]
    assert "not measured" in roofline(1e9, 1e9)["basis"]


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        occupancy(0)
    with pytest.raises(ValueError):
        roofline(1e9, 0)
    with pytest.raises(ValueError):
        lds_padding(0)

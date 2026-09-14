from __future__ import annotations

import pytest

from hipbridge import ARCHS, lds_padding, occupancy, roofline


def test_all_archs_are_64_wide_except_rdna():
    assert ARCHS["cdna3"].wavefront == 64
    assert ARCHS["rdna3"].wavefront == 32


def test_wavefront_width_comes_from_the_datasheet_rows():
    from hipbridge.analysis import wavefront_for

    for spec in ARCHS.values():
        assert wavefront_for(spec.gfx) == spec.wavefront, spec.gfx


@pytest.mark.parametrize("arch", ["gfx1030", "gfx1100", "gfx1101", "gfx1201"])
def test_every_rdna_arch_is_32_wide(arch):
    """RDNA is 32 wide whether or not a datasheet row names the card."""
    from hipbridge.analysis import wavefront_for

    assert wavefront_for(arch) == 32


@pytest.mark.parametrize("arch", ["", "gfx900", "gfx906", "gfx803", "gfx1", "gfx11000", "cdna3"])
def test_an_unknown_arch_has_no_width_rather_than_a_guess(arch):
    """The old rule said 64 for all of these.

    An empty arch is what a run without --arch passes, and a leading 9 put
    Vega-era Radeon in with CDNA. A guessed width is used to paper over a header
    skew, where a wrong value builds cleanly, so unknown has to stay unknown.
    """
    from hipbridge.analysis import wavefront_for

    assert wavefront_for(arch) is None


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

"""AMD hardware analysis. Pure arithmetic over published vendor specifications.

Every number here is a datasheet figure, NOT a measurement on your device.
Anything derived from these is a ceiling, not a result. Use hipbridge.verify to
find out what your hardware actually does.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ArchSpec:
    name: str
    gfx: str
    wavefront: int
    simds_per_cu: int
    vgprs_per_simd: int
    lds_bytes_per_cu: int
    hbm_tb_s: float
    peak_fp16_tflops: float
    source: str = "vendor datasheet"


ARCHS: dict[str, ArchSpec] = {
    "cdna3": ArchSpec("MI300X", "gfx942", 64, 4, 512, 65536, 5.3, 1307.4),
    "cdna2": ArchSpec("MI250X", "gfx90a", 64, 4, 512, 65536, 3.2, 383.0),
    "cdna1": ArchSpec("MI100", "gfx908", 64, 4, 256, 65536, 1.2, 184.6),
    "rdna3": ArchSpec("RX 7900 XTX", "gfx1100", 32, 2, 512, 131072, 0.96, 122.8),
}

# RDNA-era offload archs: gfx10xx, gfx11xx, gfx12xx. HIP compiles all of them
# 32 wide, whether or not a datasheet row above names the card.
_RDNA_ARCH = re.compile(r"gfx1[0-9a-f]{3}")


def wavefront_for(arch: str) -> int | None:
    """Wavefront width HIP compiles an offload arch to, or None when unknown.

    One owner, because three places answered this separately and all three
    guessed. They went by leading digit, gfx9 is 64 and gfx1 is 32, and fell back
    to 64 for anything else, including an empty arch. So a Vega-era Radeon
    (gfx900, gfx906) was filed with CDNA on the strength of a 9, and an RDNA run
    that did not pass --arch was handed 64. Neither fails loudly: a wrong width
    papers over a header skew with the wrong value and the build succeeds.

    Only what is known is answered: a datasheet row in ARCHS, or the RDNA family.
    Everything else is None, and callers say so rather than pick a number. The
    compiler's own __AMDGCN_WAVEFRONT_SIZE outranks this table wherever it can be
    read.
    """
    for spec in ARCHS.values():
        if spec.gfx == arch:
            return spec.wavefront
    if _RDNA_ARCH.fullmatch(arch):
        return 32
    return None


def occupancy(vgprs_per_thread: int, arch: str = "cdna3") -> dict:
    """Wavefront occupancy per SIMD, which on AMD is VGPR limited."""
    spec = ARCHS[arch]
    if vgprs_per_thread <= 0:
        raise ValueError("vgprs_per_thread must be positive")
    max_waves = 8 if spec.wavefront == 64 else 16
    waves = min(max_waves, spec.vgprs_per_simd // vgprs_per_thread)
    return {
        "arch": spec.name,
        "gfx": spec.gfx,
        "wavefronts_per_simd": waves,
        "occupancy_pct": round(100.0 * waves / max_waves, 1),
        "limited_by": "vgpr" if waves < max_waves else "none",
        "basis": "datasheet, not measured",
    }


def roofline(flops: float, bytes_moved: float, arch: str = "cdna3") -> dict:
    """Classify a kernel as compute or memory bound against the arch ceiling."""
    spec = ARCHS[arch]
    if bytes_moved <= 0:
        raise ValueError("bytes_moved must be positive")
    intensity = flops / bytes_moved
    bw = spec.hbm_tb_s * 1e12
    peak = spec.peak_fp16_tflops * 1e12
    ridge = peak / bw
    attainable = min(peak, intensity * bw)
    return {
        "arch": spec.name,
        "arithmetic_intensity": round(intensity, 4),
        "ridge_point": round(ridge, 2),
        "bound": "compute" if intensity >= ridge else "memory",
        "attainable_tflops": round(attainable / 1e12, 2),
        "peak_tflops": spec.peak_fp16_tflops,
        "basis": "datasheet ceiling, not measured",
    }


def lds_padding(row_elems: int, elem_bytes: int = 4, banks: int = 32) -> int:
    """Padding in elements to break LDS bank conflicts on a 2D shared tile."""
    if elem_bytes <= 0 or row_elems <= 0:
        raise ValueError("row_elems and elem_bytes must be positive")
    elems_per_bank_cycle = banks
    return 0 if row_elems % elems_per_bank_cycle else 1


__all__ = ["ARCHS", "ArchSpec", "lds_padding", "occupancy", "roofline", "wavefront_for"]

"""Differential verification harness. Requires the [verify] extra.

Build this before the translator, not after. It is the only thing that can tell
you a substitution is correct, and it is independently useful to anyone writing
GPU kernels by hand.

Depends on hipbridge.frontend (the IR it reads) and nothing else in core.

    from hipbridge.verify import Harness, TorchReference, shapes

    h = Harness(candidate=my_softmax,
                reference=TorchReference(lambda x: torch.softmax(x, -1)))
    print(h.run(shapes.row_wise()))
"""

from __future__ import annotations

_MISSING = "hipbridge.verify requires PyTorch.\n  pip install 'hipbridge[verify]'\n"

# name -> submodule that defines it. Mirrors each submodule's __all__ exactly,
# which tests/test_layering.py asserts in both directions.
#
# It was a hand-copied mirror with nothing keeping it in step, and it had rotted
# both ways: "BenchResult" and "Comparison" named things that have never existed
# anywhere in this repo, and since both were in __all__, `from hipbridge.verify
# import *` raised AttributeError on a machine that HAD torch installed. Missing
# in the other direction were compare.nonfinite_where_finite, which harness.py
# imports, and most of bench.
_LAZY = {
    # compare
    "Arbitration": "compare",
    "Report": "compare",
    "arbitrate": "compare",
    "check": "compare",
    "is_identity": "compare",
    "is_unwritten": "compare",
    "nonfinite_where_finite": "compare",
    "ulp_diff": "compare",
    # bench
    "LATENCY_BOUND_FRACTION": "bench",
    "LATENCY_BOUND_MIN_SHAPES": "bench",
    "Measurement": "bench",
    "ShapeRow": "bench",
    "Stat": "bench",
    "mark_latency_bound": "bench",
    "render": "bench",
    "stat_candidate": "bench",
    "stat_reference": "bench",
    "time_candidate": "bench",
    "time_reference": "bench",
    # harness
    "CaseResult": "harness",
    "Harness": "harness",
    "Summary": "harness",
    # inputs
    "DEFAULT_SWEEP": "inputs",
    "Distribution": "inputs",
    "InputSpec": "inputs",
    "Layout": "inputs",
    "NON_CONTIGUOUS": "inputs",
    "WEIGHT_FOR": "inputs",
    "generate": "inputs",
    "relayout": "inputs",
    "weight_distribution": "inputs",
    # reference
    "Availability": "reference",
    "LaunchSpec": "reference",
    "NativeReference": "reference",
    "Reference": "reference",
    "SENTINEL": "reference",
    "TorchReference": "reference",
    "hip_include_flags": "reference",
    "to_wsl_path": "reference",
    "wavefront_for": "reference",
    "wsl": "reference",
    # submodules
    "shapes": None,
}

# The submodules _LAZY re-exports from, so a test can compare the two.
_MIRRORED = ("compare", "bench", "harness", "inputs", "reference")

# Submodules that genuinely have no torch dependency and must stay importable
# with core alone. Gating these behind require() would be a lie about what they
# need, and it breaks `from hipbridge.verify import shapes` on a core install.
_TORCH_FREE = frozenset({"shapes"})


class VerifyUnavailable(ImportError):
    """Raised when the [verify] extra is not installed."""


def available() -> bool:
    """True if the [verify] extra can be imported. Never raises."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def require() -> None:
    if not available():
        raise VerifyUnavailable(_MISSING)


def __getattr__(name: str):
    # Lazy so `import hipbridge.verify` and available() work without torch.
    if name not in _LAZY:
        raise AttributeError(name)

    import importlib

    if name in _TORCH_FREE:
        return importlib.import_module(f"hipbridge.verify.{name}")

    require()
    mod = importlib.import_module(f"hipbridge.verify.{_LAZY[name] or name}")
    return mod if _LAZY[name] is None else getattr(mod, name)


def __dir__() -> list[str]:
    return sorted({*_LAZY, "available", "require", "VerifyUnavailable"})


__all__ = ["VerifyUnavailable", "available", "require", *sorted(_LAZY)]

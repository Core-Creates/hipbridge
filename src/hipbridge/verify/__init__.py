"""Differential verification harness. Requires the [verify] extra.

Build this before the translator, not after. It is the only thing that can tell
you a substitution is correct, and it is independently useful to anyone writing
GPU kernels by hand.

Promotion candidate: no import edge back into hipbridge core.

    from hipbridge.verify import Harness, TorchReference, shapes

    h = Harness(candidate=my_softmax,
                reference=TorchReference(lambda x: torch.softmax(x, -1)))
    print(h.run(shapes.row_wise()))
"""

from __future__ import annotations

_MISSING = "hipbridge.verify requires PyTorch.\n  pip install 'hipbridge[verify]'\n"

_LAZY = {
    # compare
    "Arbitration": "compare",
    "Report": "compare",
    "arbitrate": "compare",
    "check": "compare",
    "is_identity": "compare",
    "is_unwritten": "compare",
    "ulp_diff": "compare",
    # harness
    "CaseResult": "harness",
    "Harness": "harness",
    "Summary": "harness",
    # inputs
    "DEFAULT_SWEEP": "inputs",
    "Distribution": "inputs",
    "InputSpec": "inputs",
    "generate": "inputs",
    # reference
    "Availability": "reference",
    "LaunchSpec": "reference",
    "NativeReference": "reference",
    "Reference": "reference",
    "SENTINEL": "reference",
    "TorchReference": "reference",
    "hip_include_flags": "reference",
    "to_wsl_path": "reference",
    "wsl": "reference",
    # submodules
    "shapes": None,
}

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

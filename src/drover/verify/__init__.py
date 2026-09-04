"""Differential verification harness. Requires the [verify] extra.

Build this before the translator, not after. It is the only thing that can tell
you a substitution is correct, and it is independently useful to anyone writing
GPU kernels by hand.

Promotion candidate: no import edge back into drover core.
"""

from __future__ import annotations

_MISSING = "drover.verify requires PyTorch.\n  pip install 'drover[verify]'\n"


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
    # Lazy so `import drover.verify` succeeds without torch and callers can
    # check available() first.
    if name in {"compare", "ulp_diff", "Report", "check"}:
        require()
        from drover.verify import compare as _c

        return getattr(_c, name)
    raise AttributeError(name)


__all__ = ["VerifyUnavailable", "available", "check", "require"]

"""Tuned AMD Triton implementations. Requires the [kernels] extra.

Promotion candidate: this package has no import edge back into hipbridge core and
can become its own distribution unchanged.
"""

from __future__ import annotations

_MISSING = (
    "hipbridge.kernels requires Triton.\n"
    "  pip install 'hipbridge[kernels]'\n"
    "Triton publishes Linux wheels only; on Windows or macOS use a Linux host "
    "or a container for this extra."
)


class KernelsUnavailable(ImportError):
    """Raised when the [kernels] extra is not installed."""


def available() -> bool:
    """True if the [kernels] extra can be imported. Never raises."""
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def require() -> None:
    if not available():
        raise KernelsUnavailable(_MISSING)


def registry() -> dict:
    """Map Pattern -> callable returning a tuned AMD implementation."""
    require()
    from hipbridge.frontend.ir import Pattern
    from hipbridge.kernels import softmax

    return {Pattern.REDUCE_SERIAL: softmax.softmax_rowwise}


__all__ = ["KernelsUnavailable", "available", "registry", "require"]

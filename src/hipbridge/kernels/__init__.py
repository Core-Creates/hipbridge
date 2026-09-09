"""Tuned AMD Triton implementations. Requires the [kernels] extra.

Promotion candidate: this package has no import edge back into hipbridge core and
can become its own distribution unchanged.
"""

from __future__ import annotations

_MISSING = (
    "hipbridge.kernels needs Triton AND a device it can launch on.\n"
    "  pip install 'hipbridge[kernels]'\n"
    "Triton publishes Linux wheels only; on Windows or macOS use a Linux host "
    "or a container. Note that Triton installed on a CPU-only machine imports "
    "fine but cannot run: it reports 0 active drivers at launch."
)


class KernelsUnavailable(ImportError):
    """Raised when the [kernels] extra is not installed."""


def available() -> bool:
    """True if the [kernels] extra is installed AND can actually run a kernel.

    Importable is not the same as runnable. Triton installs cleanly on a
    CPU-only Linux box and then raises `0 active drivers` at first launch, so an
    import-only check reports a tuned kernel as available and blows up when it
    is called. That is the same mistake as gating a GPU probe on torch's build
    flavour: asking whether a package is present instead of whether the thing
    it needs is there.

    Never raises.
    """
    try:
        import triton  # noqa: F401
    except ImportError:
        return False

    # Triton's driver proxy raises when no backend is active. That is the
    # closest thing to "could I launch a kernel right now".
    try:
        from triton.runtime import driver

        if driver.active is not None:
            return True
    except Exception:  # noqa: BLE001 - any failure here means not runnable
        pass

    # Secondary signal. ROCm builds of torch report through the cuda namespace.
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def require() -> None:
    if not available():
        raise KernelsUnavailable(_MISSING)


# `registry()` used to live here, mapping Pattern -> implementation. It was dead
# (no call site anywhere in src/, tests/, examples/, scripts/ or the README),
# wrong (it mapped REDUCE_SERIAL to softmax, which is the wrong answer for five
# of the seven suites that share that pattern), exported in __all__ so it read
# as public API, and the only reason this package imported hipbridge.frontend
# and so contradicted the no-edge-to-core note above. A Suite now names its own
# implementation; see hipbridge.verify.suites.Suite.triton_impl.

__all__ = ["KernelsUnavailable", "available", "require"]

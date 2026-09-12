"""hipbridge: recognize CUDA kernels, substitute verified AMD implementations.

Core imports cleanly with libclang alone. The heavy pieces live behind extras
and are reached through hipbridge.kernels / hipbridge.verify, each of which exposes
available() so callers can degrade instead of crashing.

`parse_file` and `parse_source` are re-exported lazily. They come from
frontend.parser, which imports clang.cindex at module level, and importing any
submodule of a package runs that package's __init__ first. So an eager
re-export here meant `import hipbridge.verify.provenance` loaded libclang: a
dependency the [verify] extra does not declare and has no use for.

Everything else stays eager, and cheaply: analysis is pure arithmetic over
datasheet figures and frontend.ir is stdlib dataclasses.

The recognizer package is `hipbridge.recognizers`, plural, so that `recognize`
names one thing. It was `hipbridge.recognize`, which collided with the function
re-exported here under the same name: two objects, one name, one namespace, and
whichever was assigned last won. Eager assignment made the function win, so the
collision was invisible until someone made the import lazy - and then it failed
only in full runs, because a module-level __getattr__ is consulted after
__dict__, and the import system writes a submodule into its parent's __dict__ on
first load. `recognize(facts)` raised "'module' object is not callable" in a
full test run and passed in isolation.
"""

from typing import TYPE_CHECKING

from hipbridge.analysis import ARCHS, lds_padding, occupancy, roofline
from hipbridge.frontend import KernelFacts, Pattern, Recognition
from hipbridge.recognizers import recognize, registered

__version__ = "0.1.0"

# Defined in hipbridge.frontend.parser, which costs a libclang import.
_LAZY = ("parse_file", "parse_source")

if TYPE_CHECKING:  # pragma: no cover - for type checkers and editors only
    from hipbridge.frontend import parse_file, parse_source


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from hipbridge import frontend

    value = getattr(frontend, name)
    globals()[name] = value  # resolved once, then a plain attribute
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "ARCHS",
    "KernelFacts",
    "Pattern",
    "Recognition",
    "__version__",
    "lds_padding",
    "occupancy",
    "parse_file",
    "parse_source",
    "recognize",
    "registered",
    "roofline",
]

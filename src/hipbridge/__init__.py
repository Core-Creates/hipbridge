"""hipbridge: recognize CUDA kernels, substitute verified AMD implementations.

Core imports cleanly with libclang alone. The heavy pieces live behind extras
and are reached through hipbridge.kernels / hipbridge.verify, each of which exposes
available() so callers can degrade instead of crashing.

`parse_file` and `parse_source` are re-exported lazily. They come from
frontend.parser, which imports clang.cindex at module level, and importing any
submodule of a package runs that package's __init__ first. So an eager
re-export here meant `import hipbridge.verify.provenance` loaded libclang: a
dependency the [verify] extra does not declare and has no use for.

Everything else stays eager. `recognize` in particular has to be, because it
names both this package's `recognize` subpackage and the function inside it, and
a lazy __getattr__ never runs once the import system has bound the submodule as
an attribute. Eager assignment is what makes the function win, which is what it
has always done.
"""

from typing import TYPE_CHECKING

# Neither of these reaches the parser: analysis is pure arithmetic over
# datasheet figures, and frontend.ir is stdlib dataclasses.
from hipbridge.analysis import ARCHS, lds_padding, occupancy, roofline
from hipbridge.frontend import KernelFacts, Pattern, Recognition
from hipbridge.recognize import recognize, registered

__version__ = "0.1.0.dev0"

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

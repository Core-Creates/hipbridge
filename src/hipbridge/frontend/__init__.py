"""CUDA frontend: source text in, structural facts out.

The parser is imported lazily. `ir.py` is stdlib-only dataclasses, but
`parser.py` imports clang.cindex at module level, so re-exporting parse_file
eagerly meant that importing anything under `hipbridge` loaded libclang. That
reached further than it looks: `hipbridge/__init__.py` imports this package, and
importing any submodule runs the package root, so `import
hipbridge.verify.provenance` required libclang - a dependency the [verify] extra
does not declare and has no use for.

PEP 562 keeps the same public names and charges for the parser only when one of
them is touched. tests/test_layering.py asserts the property rather than the
mechanism.
"""

from typing import TYPE_CHECKING

from hipbridge.frontend.ir import (
    KernelFacts,
    Param,
    Pattern,
    Recognition,
    SharedBuffer,
)

# name -> defined in hipbridge.frontend.parser, which costs a libclang import
_LAZY = ("ParseError", "parse_file", "parse_source")

if TYPE_CHECKING:  # pragma: no cover - for type checkers and editors only
    from hipbridge.frontend.parser import ParseError, parse_file, parse_source


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from hipbridge.frontend import parser

    value = getattr(parser, name)
    globals()[name] = value  # resolved once, then a plain attribute
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "KernelFacts",
    "Param",
    "ParseError",
    "Pattern",
    "Recognition",
    "SharedBuffer",
    "parse_file",
    "parse_source",
]

"""Structural facts extracted from a CUDA kernel.

Deliberately modest. This is not a universal IR and does not claim to be one.
It records what a clang AST can tell us with certainty, and nothing else.
Fields we cannot establish are None, never a guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Pattern(str, Enum):
    """Recognized kernel shapes.

    UNKNOWN is a first-class result, not a failure. A recognizer that cannot
    identify a kernel must return UNKNOWN so the caller can fall back honestly.
    There is deliberately no default/catch-all member.
    """

    ELEMENTWISE = "elementwise"
    REDUCE_TREE = "reduce_tree"  # shared-memory tree reduction
    REDUCE_SERIAL = "reduce_serial"  # per-thread serial accumulation
    REDUCE_SHUFFLE = "reduce_shuffle"  # warp/wavefront shuffle reduction
    TRANSPOSE = "transpose"
    # A thread walks a row and writes each element from a bounded neighbourhood,
    # accumulating nothing across it. RoPE is the reason this exists: it is not
    # a reduction, and it is not flat elementwise either because one thread
    # handles a whole row and reads its neighbour. Calling it either would be a
    # lie in a field whose only job is to say what the kernel actually is.
    ROW_MAP = "row_map"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Param:
    name: str
    type: str
    is_pointer: bool
    is_const: bool


@dataclass(frozen=True)
class SharedBuffer:
    name: str
    type: str
    size_bytes: int | None


@dataclass
class KernelFacts:
    """What the parser established about one __global__ function."""

    name: str
    params: list[Param] = field(default_factory=list)
    shared: list[SharedBuffer] = field(default_factory=list)
    barriers: int = 0
    loops: int = 0
    has_halving_stride: bool = False
    shared_accumulations: int = 0
    scalar_accumulations: int = 0
    shuffle_intrinsics: list[str] = field(default_factory=list)
    atomics: list[str] = field(default_factory=list)
    uses_block_index: bool = False
    uses_thread_index: bool = False
    parse_errors: int = 0

    @property
    def pointer_params(self) -> list[Param]:
        return [p for p in self.params if p.is_pointer]

    @property
    def inputs(self) -> list[Param]:
        return [p for p in self.pointer_params if p.is_const]

    @property
    def outputs(self) -> list[Param]:
        return [p for p in self.pointer_params if not p.is_const]


@dataclass
class Recognition:
    """Result of running the recognizer registry over a kernel."""

    facts: KernelFacts
    pattern: Pattern
    confidence: str  # "certain" | "likely"
    rationale: list[str] = field(default_factory=list)

    @property
    def recognized(self) -> bool:
        return self.pattern is not Pattern.UNKNOWN

    def report(self) -> str:
        lines = [f"kernel:  {self.facts.name}", f"pattern: {self.pattern.value}"]
        if self.recognized:
            lines.append(f"confidence: {self.confidence}")
        for r in self.rationale:
            lines.append(f"  - {r}")
        if not self.recognized:
            lines.append(
                "  No substitution will be attempted. "
                "Translate this kernel by hand or extend the recognizers."
            )
        return "\n".join(lines)

"""Shape enumeration for differential testing.

Wavefront-aware by construction. AMD CDNA wavefronts are 64 wide, so shapes are
chosen to land on and around multiples of 64, not 32. The off-by-one and
off-by-63 cases are where mask errors live: a kernel that computes
`mask = offsets < n_rows` while indexing an (n_rows, n_cols) buffer passes every
square shape and fails these.
"""

from __future__ import annotations

from collections.abc import Iterator

WAVEFRONT = 64

# Row counts: one, small, exactly a wavefront, and either side of it.
_ROWS = (1, 2, 63, 64, 65, 127, 128, 1000)

# Column widths: degenerate, sub-wavefront, boundary, prime, and wide.
_COLS = (1, 2, 31, 32, 63, 64, 65, 127, 128, 257, 1024, 4096)


def row_wise() -> Iterator[tuple[int, int]]:
    """(rows, cols) pairs for row-wise kernels such as softmax or layernorm."""
    seen: set[tuple[int, int]] = set()
    for r in _ROWS:
        for c in _COLS:
            if r * c > 8_000_000:  # keep the sweep runnable on CPU
                continue
            if (r, c) not in seen:
                seen.add((r, c))
                yield (r, c)


def elementwise() -> Iterator[tuple[int]]:
    """1D sizes for elementwise kernels, clustered on wavefront boundaries."""
    sizes = (0, 1, 2, 63, 64, 65, 255, 256, 257, 1023, 1024, 1025, 65_536, 1_000_003)
    for n in sizes:
        yield (n,)


def is_wavefront_aligned(n: int) -> bool:
    return n % WAVEFRONT == 0


def sample(shapes: Iterator[tuple[int, ...]], limit: int | None = None) -> list[tuple[int, ...]]:
    """Materialize a shape iterator, optionally truncated for a fast sweep."""
    out = list(shapes)
    return out if limit is None else out[:limit]


__all__ = ["WAVEFRONT", "elementwise", "is_wavefront_aligned", "row_wise", "sample"]

"""Shape enumeration for differential testing.

Wavefront-aware by construction, for both widths AMD compiles to. CDNA
wavefronts are 64 wide and RDNA ones are 32, so shapes land on and around
multiples of each. The off-by-one and off-by-63 cases are where mask errors
live: a kernel that computes `mask = offsets < n_rows` while indexing an
(n_rows, n_cols) buffer passes every square shape and fails these.

Only 64 used to be straddled. 63, 64 and 65 do cross a multiple of 32 as well,
but the first RDNA boundary did not: no row count was 31, 32 or 33, and no
width was 33, so a mask error tied to one 32-wide wavefront had nothing here to
find it. The only device this project has been proven on is 64 wide, which is
exactly how that went unnoticed.

The order is load-bearing, not incidental. Every consumer truncates this with
`--limit`, so what a cheap run tests is decided here rather than by the caller.
"""

from __future__ import annotations

from collections.abc import Iterator

# CDNA's width, which is what is_wavefront_aligned asks about. RDNA's is 32;
# hipbridge.analysis.wavefront_for owns the answer per arch.
WAVEFRONT = 64

# Row counts: one, small, exactly a wavefront, and either side of it, for the
# 32-wide RDNA wavefront and the 64-wide CDNA one.
_ROWS = (1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 1000)

# Column widths: degenerate, either side of both wavefront widths, prime, wide,
# and wider than one block can hold. The last two exist because the kernels
# size a block as next_power_of_2(n_cols) and stopped there: a vocabulary
# softmax is 32k to 128k columns, so the most common wide-row kernel in
# inference sat outside both what the kernels supported and what the sweep
# would ever ask for.
_COLS = (1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 257, 1024, 4096, 8192, 32768)

_CAP = 8_000_000  # keep the sweep runnable on CPU

# A truncated sweep has to span both axes, and it used not to.
#
# row_wise() was a plain nested loop, rows outer and columns inner. Since every
# caller truncates with `[:limit]` and limit defaults to 12 while _COLS has 12
# entries, every shape a default run ever reached had exactly one row. The
# metered MI300X runs use limit 4, which came out as (1,1), (1,2), (1,31),
# (1,32): four shapes, one row apiece, 32 columns at the widest. With `row`
# always 0, a candidate that ignored its row stride entirely still scored
# 84/84, and so would one whose grid was sized wrong.
#
# So the first pass is chosen rather than fallen into. It covers every row
# count, and it fits inside the CLI's default --limit of 12, so a default run
# reaches all of it. The fourth shape has already brought four row counts and a
# row past the tiling threshold. Wide columns are paired with small row counts
# on purpose: (2, 4096) buys the wide-row path for 8192 elements where
# (1000, 4096) would cost four million, and a first pass nobody can afford to
# run is the same bug in a different place.
#
# The RDNA shapes come after the first eight, not before. The metered MI300X
# record is taken at --limit 4 and the nightly at --limit 2, and putting them
# in front would silently change what those runs measure. They do move RoPE's
# fourth shape: RoPE takes even widths only, so (33, 32) is its third, ahead
# of (1, 2), and (2, 2) drops out of its first four.
_FIRST_PASS = (
    (1, 1),  # degenerate: one element, no reduction to speak of
    (2, 32768),  # past the tiling threshold, at the cheapest row count
    (64, 65),  # a wavefront of rows, one column past a wavefront
    (1000, 128),  # many rows: row * stride has to hold for all of them
    (127, 63),  # one under a wavefront on both axes
    (65, 257),  # one over on rows, past 256 on columns
    (128, 31),  # aligned rows, sub-wavefront width
    (63, 1024),  # one under on rows, wide
    (32, 33),  # an RDNA wavefront of rows, one column past one
    (33, 32),  # one row past an RDNA wavefront, columns on one; RoPE can take it
    (31, 31),  # one under an RDNA wavefront on both axes
)


def row_wise() -> Iterator[tuple[int, int]]:
    """(rows, cols) pairs for row-wise kernels such as softmax or layernorm.

    Yields the hand-picked first pass, then the rest of the product along
    ascending diagonals so later prefixes keep mixing both axes instead of
    walking one row count to exhaustion. The full sweep is the same set of
    shapes it has always been; only the order changed.
    """
    seen: set[tuple[int, int]] = set()

    def fresh(r: int, c: int) -> bool:
        if r * c > _CAP or (r, c) in seen:
            return False
        seen.add((r, c))
        return True

    for r, c in _FIRST_PASS:
        if fresh(r, c):
            yield (r, c)

    rest = [(r, c) for r in _ROWS for c in _COLS]
    rest.sort(key=lambda rc: (_ROWS.index(rc[0]) + _COLS.index(rc[1]), _ROWS.index(rc[0])))
    for r, c in rest:
        if fresh(r, c):
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

#!/usr/bin/env python3
"""Fail when a kernel got slower, without failing when a clock wobbled.

The nightly verifies and does not time, so a substitution could lose a third of
its throughput and every check would stay green until somebody read a table. The
committed benchmark is right there and nothing compares against it.

The hard part is not the comparison, it is not crying wolf. Two runs of
identical code on this MI300X measured row_softmax at 5.0x and 4.5x slower than
tuned HIP at 1x1024, because that shape is dispatch-dominated - 4us against 20us
- so a microsecond of jitter is a quarter of the ratio. A flat "fail above 10%"
would fire on noise within a week and be switched off in two.

So a regression has to clear two bars at once:

  1. the median moved by more than the threshold, and
  2. the new fastest run is slower than the old slowest run

The second is the one that matters. Every row carries [min-max] across repeated
runs, which is a measured noise band, and when the bands still overlap the two
numbers have not been shown to differ at all. Both conditions, or it is noise.

Lives in scripts/ deliberately. hipbridge.verify is hashed into every report's
provenance digest, so a comparator living there would invalidate the reports it
exists to compare, and each round of that costs a GPU run. This reads reports;
it is not measured code.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

Row = tuple[str, str, str]  # kernel, dtype, shape


def parse(text: str) -> dict[Row, tuple[float, float, float]]:
    """Candidate median, min and max per kernel, dtype and shape."""
    out: dict[Row, tuple[float, float, float]] = {}
    kernel = dtype = None
    for line in text.splitlines():
        head = re.match(r"## (\w+) \((\w+)\)", line)
        if head:
            kernel, dtype = head.group(1), head.group(2)
            continue
        shape = re.match(r"\s*(\d+x\d+)\s", line)
        if not (kernel and shape):
            continue
        # Four columns of `median [min-max]`: original, tuned HIP, torch, candidate.
        cells = re.findall(r"([\d.]+) \[([\d.]+)-([\d.]+)\]", line)
        if len(cells) < 4:
            continue
        median, low, high = cells[3]
        out[(kernel, dtype, shape.group(1))] = (float(median), float(low), float(high))
    return out


def regressions(before: dict, after: dict, threshold: float) -> list[str]:
    findings = []
    for key in sorted(before.keys() & after.keys()):
        (old_med, _, old_max), (new_med, new_min, _) = before[key], after[key]
        if old_med <= 0:
            continue
        ratio = new_med / old_med
        if ratio <= 1 + threshold:
            continue
        if new_min <= old_max:
            # The bands overlap, so these two numbers have not been shown to
            # differ. Reported, because a run of these is worth a human glance
            # even when no single one is conclusive.
            findings.append(
                f"  noted   {'/'.join(key):40s} {old_med:8.1f} -> {new_med:8.1f} us "
                f"({ratio:.2f}x, bands overlap)"
            )
            continue
        findings.append(
            f"  SLOWER  {'/'.join(key):40s} {old_med:8.1f} -> {new_med:8.1f} us "
            f"({ratio:.2f}x, {new_min:.1f} > {old_max:.1f})"
        )
    return findings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("before", type=Path, help="the committed benchmark")
    ap.add_argument("after", type=Path, help="the benchmark just measured")
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="fractional slowdown of the median before a row is considered (default 0.15)",
    )
    args = ap.parse_args(argv)

    before = parse(args.before.read_text(encoding="utf-8"))
    after = parse(args.after.read_text(encoding="utf-8"))
    if not before or not after:
        print(f"nothing to compare: {len(before)} rows before, {len(after)} after")
        return 0

    shared = before.keys() & after.keys()
    print(f"comparing {len(shared)} rows measured in both, threshold {args.threshold:.0%}")

    # A row that vanished is not a regression, but it is a change worth naming:
    # a kernel dropped from the sweep stops being watched by this check.
    for gone in sorted(before.keys() - after.keys()):
        print(f"  gone    {'/'.join(gone)}")

    findings = regressions(before, after, args.threshold)
    real = [f for f in findings if "SLOWER" in f]
    for line in findings:
        print(line)

    if real:
        print(f"\n{len(real)} kernel(s) measurably slower than the committed benchmark")
        return 1
    print("\nno regression outside the measured noise bands")
    return 0


if __name__ == "__main__":
    sys.exit(main())

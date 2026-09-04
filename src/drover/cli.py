"""Command line interface.

Note the import discipline: drover.kernels and drover.verify are imported
*inside* functions, never at module level, so the CLI works with core alone.
tests/test_layering.py enforces this.
"""

from __future__ import annotations

import argparse
import sys

from drover import __version__
from drover.analysis import ARCHS, occupancy, roofline
from drover.frontend import parse_file
from drover.recognize import recognize, registered


def _cmd_inspect(args) -> int:
    kernels = parse_file(args.file)
    if not kernels:
        print(f"no kernel definitions found in {args.file}", file=sys.stderr)
        return 1
    for facts in kernels:
        result = recognize(facts)
        print(result.report())
        print()
    return 0


def _cmd_analyze(args) -> int:
    print("occupancy:", occupancy(args.vgprs, args.arch))
    if args.flops and args.bytes:
        print("roofline: ", roofline(args.flops, args.bytes, args.arch))
    return 0


def _cmd_info(args) -> int:
    print(f"drover {__version__}")
    print(f"recognizers: {', '.join(registered())}")
    print(f"architectures: {', '.join(ARCHS)}")

    # Lazy, guarded: core must run without either extra installed.
    from drover import kernels, verify

    print(f"[kernels] extra: {'available' if kernels.available() else 'not installed'}")
    print(f"[verify]  extra: {'available' if verify.available() else 'not installed'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="drover", description=__doc__)
    p.add_argument("--version", action="version", version=f"drover {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    insp = sub.add_parser("inspect", help="parse a .cu file and report recognized kernels")
    insp.add_argument("file")
    insp.set_defaults(fn=_cmd_inspect)

    ana = sub.add_parser("analyze", help="occupancy and roofline against datasheet ceilings")
    ana.add_argument("--vgprs", type=int, default=48)
    ana.add_argument("--arch", default="cdna3", choices=sorted(ARCHS))
    ana.add_argument("--flops", type=float)
    ana.add_argument("--bytes", type=float)
    ana.set_defaults(fn=_cmd_analyze)

    info = sub.add_parser("info", help="show capabilities and which extras are installed")
    info.set_defaults(fn=_cmd_info)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())

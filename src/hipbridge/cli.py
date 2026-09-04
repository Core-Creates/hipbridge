"""Command line interface.

Note the import discipline: hipbridge.kernels and hipbridge.verify are imported
*inside* functions, never at module level, so the CLI works with core alone.
tests/test_layering.py enforces this.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hipbridge import __version__
from hipbridge.analysis import ARCHS, occupancy, roofline
from hipbridge.frontend import parse_file
from hipbridge.recognize import recognize, registered


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


def _cmd_verify(args) -> int:
    """Run the example suites against a real device.

    Exits 0 and reports a skip when no device is present, so this is safe to
    wire into CI that usually has no GPU. Pass --require to make absence fatal.
    """
    # Lazy, guarded: core must run without the [verify] extra installed.
    from hipbridge import verify

    if not verify.available():
        print("SKIP: [verify] extra not installed (pip install 'hipbridge[verify]')")
        return 1 if args.require else 0

    from hipbridge.verify import suites

    examples = Path(args.examples)
    prefix = verify.wsl(args.wsl) if args.wsl else []
    lines: list[str] = []
    failed = False
    ran = 0

    for suite in suites.BUILTIN:
        ref = verify.NativeReference(
            source=suite.source(examples),
            launch=suite.launch,
            toolchain=args.toolchain,
            command_prefix=prefix,
            extra_flags=(["--offload-arch=" + args.arch] if args.arch else []),
        )
        avail = ref.availability()
        if not avail:
            msg = f"SKIP  {suite.name}: {avail.reason}"
            print(msg)
            lines.append(msg)
            continue

        candidate, described = suites.candidate_for(suite)
        print(f"running {suite.name} against {args.toolchain} device")
        print(f"  candidate: {described}")
        shape_list = list(suite.shapes())[: args.limit]

        summary = verify.Harness(
            candidate=candidate,
            reference=ref,
            oracle=suite.oracle,
            name=f"{suite.name} vs original on {args.toolchain}",
        ).run(shape_list)
        ran += 1
        print(summary)
        lines.append(str(summary))
        failed |= not summary.ok

    if args.report:
        Path(args.report).write_text(
            "# hipbridge verification report\n\n"
            f"toolchain: `{args.toolchain}`  arch: `{args.arch or 'default'}`\n\n"
            + "\n\n".join(f"```\n{x}\n```" for x in lines)
            + "\n",
            encoding="utf-8",
        )
        print(f"\nreport written to {args.report}")

    if ran == 0:
        print("no suite ran (no device available)")
        return 1 if args.require else 0
    return 1 if failed else 0


def _cmd_bench(args) -> int:
    """Time the candidate against the original on device.

    Refuses to report a speedup for a shape that failed verification. A fast
    wrong kernel is not a result, and a benchmark that does not say whether the
    numbers were correct is the kind this project exists to distrust.
    """
    from hipbridge import verify

    if not verify.available():
        print("SKIP: [verify] extra not installed")
        return 1 if args.require else 0

    import torch

    from hipbridge.verify import bench, suites

    examples = Path(args.examples)
    prefix = verify.wsl(args.wsl) if args.wsl else []
    shapes = [tuple(int(d) for d in s.split("x")) for s in args.shapes.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    failed = False
    if device != "cuda":
        print("WARNING: no GPU visible to torch, so the candidate runs on the host")
        print("         while the original runs on device. Ratios are suppressed as")
        print("         NOT COMPARABLE. On Windows this is normally a CPU-only torch")
        print("         wheel; Triton has no Windows build either.")
        print()

    for suite in suites.BUILTIN:
        ref = verify.NativeReference(
            source=suite.source(examples),
            launch=suite.launch,
            toolchain=args.toolchain,
            command_prefix=prefix,
            extra_flags=(["--offload-arch=" + args.arch] if args.arch else []),
        )
        avail = ref.availability()
        if not avail:
            print(f"SKIP  {suite.name}: {avail.reason}")
            continue

        candidate, described = suites.candidate_for(suite)
        print(f"{suite.name} on {args.toolchain}, device={device}, reps={args.reps}")
        print(f"  candidate: {described}")
        print(f"  original : {suite.source_file} compiled with {args.toolchain}")
        print()

        for shape in shapes:
            x = verify.generate(verify.InputSpec(shape=shape), device=device)

            # Verify at this exact shape before timing it.
            summary = verify.Harness(
                candidate=candidate,
                reference=ref,
                oracle=suite.oracle,
                name=f"{suite.name}@{shape}",
                distributions=(verify.Distribution.NORMAL,),
            ).run([shape])
            ok = summary.ok
            failed |= not ok

            try:
                print(bench.compare(candidate, ref, x, reps=args.reps, verified=ok))
            except RuntimeError as exc:
                print(f"  timing unavailable at {shape}: {exc}")
                failed = True

    return 1 if (failed and args.require) else 0


def _cmd_info(args) -> int:
    print(f"hipbridge {__version__}")
    print(f"recognizers: {', '.join(registered())}")
    print(f"architectures: {', '.join(ARCHS)}")

    # Lazy, guarded: core must run without either extra installed.
    from hipbridge import kernels, verify

    print(f"[kernels] extra: {'available' if kernels.available() else 'not installed'}")
    print(f"[verify]  extra: {'available' if verify.available() else 'not installed'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hipbridge", description=__doc__)
    p.add_argument("--version", action="version", version=f"hipbridge {__version__}")
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

    ver = sub.add_parser("verify", help="run example suites against a real device")
    ver.add_argument("--toolchain", default="hipcc", choices=["hipcc", "nvcc"])
    ver.add_argument("--arch", default="", help="offload arch, e.g. gfx942 for MI300X")
    ver.add_argument("--limit", type=int, default=12, help="max shapes per suite (cost control)")
    ver.add_argument("--examples", default="examples", help="directory holding the .cu files")
    ver.add_argument("--wsl", default="", metavar="DISTRO", help="run the toolchain inside WSL2")
    ver.add_argument("--report", default="", help="write a markdown report to this path")
    ver.add_argument(
        "--require",
        action="store_true",
        help="fail instead of skipping when no device is available",
    )
    ver.set_defaults(fn=_cmd_verify)

    ben = sub.add_parser("bench", help="time the candidate against the original on device")
    ben.add_argument("--toolchain", default="hipcc", choices=["hipcc", "nvcc"])
    ben.add_argument("--arch", default="", help="offload arch, e.g. gfx942")
    ben.add_argument("--shapes", default="1x1024,64x1024,1024x1024,4096x4096")
    ben.add_argument("--reps", type=int, default=100)
    ben.add_argument("--examples", default="examples")
    ben.add_argument("--wsl", default="", metavar="DISTRO")
    ben.add_argument("--require", action="store_true")
    ben.set_defaults(fn=_cmd_bench)

    info = sub.add_parser("info", help="show capabilities and which extras are installed")
    info.set_defaults(fn=_cmd_info)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())

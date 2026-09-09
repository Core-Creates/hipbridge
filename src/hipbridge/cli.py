"""Command line interface.

Note the import discipline: hipbridge.kernels and hipbridge.verify are imported
*inside* functions, never at module level, so the CLI works with core alone.
tests/test_layering.py enforces this.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
from pathlib import Path

from hipbridge import __version__
from hipbridge.analysis import ARCHS, occupancy, roofline
from hipbridge.frontend import parse_file
from hipbridge.recognizers import recognize, registered

# The precisions --dtype accepts. Spelled here as strings so argparse can offer
# them without importing torch, and asserted against verify.pipeline.DTYPES by
# tests/test_verify_cli.py: the set used to be written out in three places and
# adding one meant finding all of them.
DTYPE_CHOICES = ["float32", "float16", "bfloat16"]


def _dtypes(args) -> tuple:
    """Precisions to sweep, from repeated --dtype flags.

    Empty means float32 alone, which is what the harness defaults to. Named
    here rather than deep in the harness so a CI run can ask for the precision
    it cares about without a code change.
    """
    from hipbridge.verify.pipeline import dtypes_by_name

    try:
        return dtypes_by_name(getattr(args, "dtype", None))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _report_path(args, command: str) -> str:
    """Where a report goes: an explicit path, or the tracked results directory.

    `--report` with no value means "save this where results are kept", which is
    the common case and the one worth making frictionless. Filenames are
    deterministic so a re-run shows up as a diff rather than as a new file that
    nobody compares against the last one.
    """
    from hipbridge.verify import provenance

    if args.report == "AUTO":
        return provenance.default_path(command, args.toolchain, args.arch)
    return args.report


# Exit codes, which are the machine-readable half of this CLI's contract.
#
# A pipeline reads $?, not prose, and these used to conflate the two outcomes
# that matter most. `port` returned 0 both when it proved a substitution and
# when it could not prove one, so a job on a machine without hipcc read
# "unproven" as success while the text on screen said the opposite. Its own
# docstring had promised 5 for that since it was written.
#
# The distinction that matters is between a claim that failed and a claim that
# was never tested. Both are "not proved" and they call for opposite responses:
# one is a defect to go fix, the other a machine to go find.
EXIT_OK = 0  # the claim held: proved, verified, or reported
EXIT_USAGE = 1  # bad input: no such file, or no kernel definitions in it
EXIT_REFUSED = 2  # refused to act: generated code without --allow-untrusted-code
EXIT_NOTHING = 3  # nothing to claim: unrecognized, or no substitution proposed
EXIT_REJECTED = 4  # the claim was tested and did not survive it
EXIT_UNPROVABLE = 5  # the claim could not be tested: no toolchain, device or extra


def _record(args, **fields) -> None:
    """Accumulate the machine-readable result for --json.

    Commands build this as they go rather than returning it, because they
    already return an exit code and the two answer different questions: the code
    says what happened, this says what was measured.
    """
    payload = getattr(args, "_payload", None)
    if payload is None:
        payload = {}
        args._payload = payload
    payload.update(fields)


def _json_safe(value):
    """Replace non-finite floats with null, recursively.

    An error of `inf` is a real outcome here: it is what a candidate scores when
    it returns NaN where the oracle returns a number. But `Infinity` and `NaN`
    are not JSON, only a Python convention that json.dumps emits by default and
    strict parsers reject. Emitting null keeps the document parseable; the
    verdict beside it already says what happened.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _parse(path: str):
    """Parse a .cu file, reporting an unreadable one rather than raising.

    `hipbridge inspect typo.cu` exited with a FileNotFoundError traceback, which
    is Python's exit 1 by accident rather than this CLI's EXIT_USAGE on purpose.
    A traceback is not a diagnosis, and an exit code arrived at by accident is
    not a contract.
    """
    try:
        return parse_file(path), None
    except OSError as exc:
        print(f"cannot read {path}: {exc.strerror or exc}", file=sys.stderr)
        return None, EXIT_USAGE


def _cmd_inspect(args) -> int:
    kernels, problem = _parse(args.file)
    if problem:
        return problem
    if not kernels:
        print(f"no kernel definitions found in {args.file}", file=sys.stderr)
        return EXIT_USAGE
    seen = []
    for facts in kernels:
        result = recognize(facts)
        seen.append(result.as_dict())
        print(result.report())
        print()
    _record(args, file=args.file, kernels=seen)
    return EXIT_OK


def _cmd_analyze(args) -> int:
    print("occupancy:", occupancy(args.vgprs, args.arch))
    if args.flops and args.bytes:
        print("roofline: ", roofline(args.flops, args.bytes, args.arch))
    return EXIT_OK


def _device(args):
    """The compile-and-run target these flags describe."""
    from hipbridge.verify.pipeline import Device

    return Device(
        toolchain=args.toolchain,
        arch=args.arch,
        wsl=getattr(args, "wsl", ""),
        examples=Path(getattr(args, "examples", "examples")),
    )


def _sweep(args):
    """What to sweep over, beyond the shapes each suite declares."""
    from hipbridge.verify.pipeline import Sweep

    return Sweep(
        dtypes=_dtypes(args),
        limit=getattr(args, "limit", None),
        layouts=bool(getattr(args, "layouts", False)),
    )


# Outcome is the library's vocabulary; these are this CLI's contract with a
# shell. Kept as a mapping rather than pushed into the enum, because an exit
# code is a fact about a process and means nothing to a caller in Python.
_EXIT_FOR = {
    "ok": EXIT_OK,
    "usage": EXIT_USAGE,
    "refused": EXIT_REFUSED,
    "nothing": EXIT_NOTHING,
    "rejected": EXIT_REJECTED,
    "unprovable": EXIT_UNPROVABLE,
}


def _write_report(args, command: str, title: str, body: str) -> None:
    from hipbridge.verify import provenance

    written = provenance.write(_report_path(args, command), title, args.toolchain, args.arch, body)
    print(f"report written to {written}")


def _cmd_verify(args) -> int:
    """Run the example suites against a real device.

    Exits 0 and reports a skip when no device is present, so this is safe to
    wire into CI that usually has no GPU. Pass --require to make absence fatal.
    """
    # Lazy, guarded: core must run without the [verify] extra installed.
    from hipbridge import verify

    if not verify.available():
        print("SKIP: [verify] extra not installed (pip install 'hipbridge[verify]')")
        return EXIT_UNPROVABLE if args.require else EXIT_OK

    from hipbridge.verify import pipeline

    run = pipeline.run_suites(_device(args), _sweep(args), progress=print)

    if args.report:
        print()
        _write_report(args, "verify", "hipbridge verification", run.report_body())

    _record(args, **run.as_dict())

    if run.ran == 0:
        print("no suite ran (no device available)")
        return EXIT_UNPROVABLE if args.require else EXIT_OK
    return EXIT_OK if run.ok else EXIT_REJECTED


def _cmd_bench(args) -> int:
    """Time the candidate against the original on device.

    Refuses to report a speedup for a shape that failed verification. A fast
    wrong kernel is not a result, and a benchmark that does not say whether the
    numbers were correct is the kind this project exists to distrust.
    """
    from hipbridge import verify

    if not verify.available():
        print("SKIP: [verify] extra not installed")
        return EXIT_UNPROVABLE if args.require else EXIT_OK

    from hipbridge.verify import pipeline

    shapes = [tuple(int(d) for d in s.split("x")) for s in args.shapes.split(",")]
    run = pipeline.bench_suites(
        _device(args),
        _sweep(args),
        shapes,
        reps=args.reps,
        runs=args.runs,
        progress=print,
    )

    if args.report and run.sections:
        print()
        _write_report(args, "bench", "hipbridge benchmark", run.report_body())

    _record(args, **run.as_dict())
    return _EXIT_FOR[run.outcome.value]


def _cmd_port(args) -> int:
    """recognize -> propose a substitute -> prove it on device -> emit the code.

    The whole pipeline in one command, and the only one of them that answers the
    question a user actually has: can I replace this kernel, and how do I know.

    Exit codes are the interface here: 0 substituted and proved, 3 recognized but
    nothing to propose, 4 proposed but the proof failed, 5 no device to prove it
    on. A substitution is never recommended on the strength of recognition alone.
    """
    from hipbridge import verify

    if not verify.available():
        # Recognition still runs: it needs no extra, and it is the half of the
        # answer this machine can give.
        kernels, problem = _parse(args.file)
        if problem:
            return problem
        if not kernels:
            print(f"no kernel definitions found in {args.file}", file=sys.stderr)
            return EXIT_USAGE
        result = recognize(kernels[0])
        _record(args, file=args.file, recognition=result.as_dict(), proved=False)
        print(result.report())
        print()
        print("SKIP: [verify] extra not installed, so no substitution can be proved")
        return EXIT_UNPROVABLE

    from hipbridge.verify import pipeline

    outcome = pipeline.port_file(
        args.file,
        _device(args),
        _sweep(args),
        kernel=args.kernel,
        eps=args.eps,
        block=int(args.block) if args.block else None,
        progress=print,
    )

    payload = outcome.as_dict()
    payload["file"] = args.file
    _record(args, **payload)

    if outcome.error:
        print(outcome.error, file=sys.stderr)
        return _EXIT_FOR[outcome.outcome.value]

    if outcome.outcome is pipeline.Outcome.NOTHING:
        print("no substitution proposed.")
        print()
        # A near miss is worth explaining. Without this, a kernel that matched
        # everything except the order of its weights reports the same "nothing
        # to propose" as a kernel nobody recognised, and sends its author
        # looking for a missing feature instead of reading their signature.
        for note in outcome.declined:
            print(f"  {note}")
        if outcome.declined:
            print()
        if outcome.recognition.recognized:
            print(
                f"  {outcome.facts.name} is a {outcome.recognition.pattern.value}, "
                "but a pattern is not a"
            )
            print("  licence to substitute: several different computations share it.")
            print("  Nothing here matched this kernel's maths closely enough to try.")
        else:
            print("  no recognizer claimed this kernel, so there is nothing to propose.")
        print()
        print("  Translate it by hand, or extend hipbridge.verify.substitutions.")
        return EXIT_NOTHING

    if outcome.outcome is pipeline.Outcome.UNPROVABLE:
        print(f"CANNOT PROVE: {outcome.unprovable}")
        print()
        print("  The substitution is unproven, so it is not recommended. Re-run this")
        print("  on a machine with the toolchain and a device.")
        return EXIT_UNPROVABLE

    if not outcome.proved:
        # The probe now lives in Harness, so verify, bench and synth get it too.
        # What stays here is the advice only port can give: it is the command
        # that inferred the launch geometry, so it is the one that can suggest
        # --block.
        if outcome.summary.probe_failure:
            print(f"  Inferred block={outcome.launch.block}; override it with --block.")
            print("  No substitution is claimed, because being better than a broken")
            print("  reference is not evidence of anything.")
            return EXIT_REJECTED
        print("SUBSTITUTION REJECTED. The proposal did not survive verification,")
        print("which is the system working: recognition proposes, the oracle decides.")
        return EXIT_REJECTED

    suite = outcome.proposal.suite
    print("SUBSTITUTION PROVED. Use it like this:")
    print()
    print(f"    {suite.usage_import}")
    print()
    print(f"    {suite.usage_call}   # replaces {outcome.facts.name}")
    print()
    print(
        f"Proved on {args.toolchain}"
        f"{' for ' + args.arch if args.arch else ''} against your own kernel,"
    )
    print("judged against a float64 oracle rather than against the original's rounding.")
    if args.report:
        _write_report(
            args, f"port-{outcome.facts.name}", "hipbridge port report", outcome.report_body()
        )
    return EXIT_OK


def _cmd_synth(args) -> int:
    """Ask a generator for a kernel, then let the oracle decide.

    The same proof `port` uses, pointed at code nobody has read. That is the
    point: coverage grows one hand-written Triton kernel at a time, and a model
    can write those quickly, but only if something trustworthy decides which to
    keep. This repository already is that something.

    Nothing is recommended on the strength of having been generated. A candidate
    that passes has matched the caller's compiled kernel across the same sweep
    every shipped kernel had to pass; one that fails is recorded and discarded.
    """
    from hipbridge import synth, verify

    if not verify.available():
        print("SKIP: [verify] extra not installed, so nothing could be proved")
        return EXIT_UNPROVABLE

    from hipbridge.verify import pipeline

    generator = (
        synth.ScriptGenerator(command=args.generator.split())
        if args.generator
        else synth.StaticGenerator(
            sources=[Path(p).read_text(encoding="utf-8") for p in args.file_candidate]
        )
    )

    outcome = pipeline.synth_file(
        args.file,
        _device(args),
        _sweep(args),
        generator,
        count=args.count,
        allow_untrusted_code=args.allow_untrusted_code,
        progress=print,
        show_prompt=args.show_prompt,
    )

    if outcome.outcome is pipeline.Outcome.USAGE:
        print(outcome.error, file=sys.stderr)
        return EXIT_USAGE

    if outcome.outcome is pipeline.Outcome.NOTHING:
        print(f"{outcome.error}.")
        print("Generation without a proof is not something this tool will do.")
        return EXIT_NOTHING

    if outcome.outcome is pipeline.Outcome.UNPROVABLE:
        print(f"CANNOT PROVE: {outcome.error}")
        print("Generated code is worth nothing unproved, so nothing is generated.")
        return EXIT_UNPROVABLE

    if outcome.outcome is pipeline.Outcome.REFUSED:
        print(f"REFUSED: {outcome.error}")
        print()
        print("  Generated kernels execute in this process with its privileges.")
        print("  Pass --allow-untrusted-code once you are on a machine where that")
        print("  is acceptable, which is not one holding credentials you care about.")
        return EXIT_REFUSED

    print(outcome.found)
    winner = outcome.winner
    if winner is None:
        return EXIT_REJECTED

    print()
    print(winner.summary)
    if args.out:
        Path(args.out).write_text(winner.candidate.source, encoding="utf-8")
        print(f"kernel written to {args.out}")
    return EXIT_OK


def _cmd_info(args) -> int:
    print(f"hipbridge {__version__}")
    print(f"recognizers: {', '.join(registered())}")
    print(f"architectures: {', '.join(ARCHS)}")

    # Lazy, guarded: core must run without either extra installed.
    from hipbridge import kernels, verify

    print(f"[kernels] extra: {'available' if kernels.available() else 'not installed'}")
    print(f"[verify]  extra: {'available' if verify.available() else 'not installed'}")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hipbridge", description=__doc__)
    p.add_argument("--version", action="version", version=f"hipbridge {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    insp = sub.add_parser("inspect", help="parse a .cu file and report recognized kernels")
    insp.add_argument("file")
    insp.add_argument(
        "--json",
        action="store_true",
        help="emit the result as JSON on stdout, in place of the prose report",
    )
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
    ver.add_argument(
        "--dtype",
        action="append",
        default=[],
        choices=DTYPE_CHOICES,
        help="precision to sweep; repeatable, defaults to float32",
    )
    ver.add_argument(
        "--report",
        nargs="?",
        const="AUTO",
        default="",
        metavar="PATH",
        help="write a markdown report; bare flag saves under results/",
    )
    ver.add_argument(
        "--require",
        action="store_true",
        help="fail instead of skipping when no device is available",
    )
    ver.add_argument(
        "--no-layouts",
        dest="layouts",
        action="store_false",
        help="skip the non-contiguous layout pass (transposed, sliced, padded inputs)",
    )
    ver.add_argument(
        "--json",
        action="store_true",
        help="emit the result as JSON on stdout, in place of the prose report",
    )
    ver.set_defaults(fn=_cmd_verify)

    ben = sub.add_parser("bench", help="time the candidate against the original on device")
    ben.add_argument("--toolchain", default="hipcc", choices=["hipcc", "nvcc"])
    ben.add_argument("--arch", default="", help="offload arch, e.g. gfx942")
    ben.add_argument("--shapes", default="1x1024,64x1024,1024x1024,4096x4096")
    ben.add_argument("--reps", type=int, default=100)
    ben.add_argument(
        "--runs",
        type=int,
        default=5,
        help="repeat each timing this many times; the table reports median [min-max]",
    )
    ben.add_argument("--examples", default="examples")
    ben.add_argument("--wsl", default="", metavar="DISTRO")
    ben.add_argument(
        "--dtype",
        action="append",
        default=[],
        choices=DTYPE_CHOICES,
        help="precision to sweep; repeatable, defaults to float32",
    )
    ben.add_argument(
        "--report",
        nargs="?",
        const="AUTO",
        default="",
        metavar="PATH",
        help="write a markdown report; bare flag saves under results/",
    )
    ben.add_argument("--require", action="store_true")
    ben.add_argument(
        "--json",
        action="store_true",
        help="emit the result as JSON on stdout, in place of the prose tables",
    )
    ben.set_defaults(fn=_cmd_bench)

    prt = sub.add_parser("port", help="recognize, substitute and prove, in one step")
    prt.add_argument("file")
    prt.add_argument("--kernel", default="", help="which kernel, if the file has several")
    prt.add_argument("--toolchain", default="hipcc", choices=["hipcc", "nvcc"])
    prt.add_argument("--arch", default="", help="offload arch, e.g. gfx942 for MI300X")
    prt.add_argument("--limit", type=int, default=12, help="max shapes to prove over")
    prt.add_argument(
        "--eps",
        type=float,
        default=None,
        help="the epsilon this kernel normalises with, when it is not a readable literal",
    )
    prt.add_argument(
        "--block",
        type=int,
        default=0,
        help="threads per block for the original; inferred from the source when unset",
    )
    prt.add_argument("--wsl", default="", metavar="DISTRO")
    prt.add_argument(
        "--report",
        nargs="?",
        const="AUTO",
        default="",
        metavar="PATH",
        help="write a markdown report; bare flag saves under results/",
    )
    prt.add_argument(
        "--no-layouts",
        dest="layouts",
        action="store_false",
        help="skip the non-contiguous layout pass (transposed, sliced, padded inputs)",
    )
    prt.add_argument(
        "--json",
        action="store_true",
        help="emit the result as JSON on stdout, in place of the prose report",
    )
    prt.set_defaults(fn=_cmd_port)

    syn = sub.add_parser("synth", help="generate candidate kernels and prove them on device")
    syn.add_argument("file")
    syn.add_argument("--toolchain", default="hipcc", choices=["hipcc", "nvcc"])
    syn.add_argument("--arch", default="", help="offload arch, e.g. gfx942 for MI300X")
    syn.add_argument("--count", type=int, default=5, help="how many candidates to ask for")
    syn.add_argument("--limit", type=int, default=12, help="max shapes to prove over")
    syn.add_argument("--wsl", default="", metavar="DISTRO")
    syn.add_argument(
        "--generator",
        default="",
        metavar="CMD",
        help="command that prints a Triton kernel on stdout, given the prompt on stdin",
    )
    syn.add_argument(
        "--file-candidate",
        action="append",
        default=[],
        metavar="PATH",
        help="propose this file instead of calling a generator; repeatable",
    )
    syn.add_argument("--out", default="", help="write the surviving kernel here")
    syn.add_argument("--show-prompt", action="store_true", help="print the prompt and continue")
    syn.add_argument(
        "--allow-untrusted-code",
        action="store_true",
        help="execute generated code in this process; it is not sandboxed",
    )
    syn.set_defaults(fn=_cmd_synth)

    info = sub.add_parser("info", help="show capabilities and which extras are installed")
    info.set_defaults(fn=_cmd_info)

    args = p.parse_args(argv)
    if not getattr(args, "json", False):
        return args.fn(args)

    # The prose and the JSON are alternatives, not companions: a caller parsing
    # stdout should not have to skip past a report first. Diagnostics still go
    # to stderr, where they were already going.
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = args.fn(args)

    payload = getattr(args, "_payload", None) or {}
    payload = {"command": args.cmd, **payload, "exit_code": code}
    print(json.dumps(_json_safe(payload), indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())

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


def _dtypes(args) -> tuple:
    """Precisions to sweep, from repeated --dtype flags.

    Empty means float32 alone, which is what the harness defaults to. Named
    here rather than deep in the harness so a CI run can ask for the precision
    it cares about without a code change.
    """
    import torch

    known = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    out = []
    for name in args.dtype or []:
        if name not in known:
            raise SystemExit(f"unknown dtype {name}; choose from {', '.join(known)}")
        out.append(known[name])
    return tuple(out)


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
            extras=tuple(o.build for o in suite.extras),
            dtypes=_dtypes(args),
            name=f"{suite.name} vs original on {args.toolchain}",
        ).run(shape_list)
        ran += 1
        print(summary)
        lines.append(str(summary))
        failed |= not summary.ok

    if args.report:
        from hipbridge.verify import provenance

        body = "\n\n".join("```\n" + x + "\n```" for x in lines)
        written = provenance.write(
            _report_path(args, "verify"), "hipbridge verification", args.toolchain, args.arch, body
        )
        print(f"\nreport written to {written}")

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
    report_sections: list[str] = []
    if device != "cuda":
        print("WARNING: no GPU visible to torch, so the candidate runs on the host")
        print("         while the original runs on device. Ratios are suppressed as")
        print("         NOT COMPARABLE. On Windows this is normally a CPU-only torch")
        print("         wheel; Triton has no Windows build either.")
        print()

    for suite in suites.BUILTIN:
        flags = ["--offload-arch=" + args.arch] if args.arch else []

        # Every baseline gets the same toolchain, the same flags and the same
        # timing path. The first is the original as written; the rest exist so a
        # large ratio against a naive kernel cannot pass for the whole story.
        refs = []
        for base in suite.all_baselines():
            r = verify.NativeReference(
                source=base.source(examples),
                launch=base.launch,
                toolchain=args.toolchain,
                command_prefix=prefix,
                extra_flags=flags,
            )
            avail = r.availability()
            if not avail:
                print(f"SKIP  {suite.name} [{base.name}]: {avail.reason}")
                continue
            refs.append((base, r))
        if not refs:
            continue

        candidate, described = suites.candidate_for(suite)
        print(
            f"{suite.name} on {args.toolchain}, device={device}, reps={args.reps}, runs={args.runs}"
        )
        print(f"  candidate: {described}")
        for base, _ in refs:
            print(f"  {base.name:<9}: {base.source_file} compiled with {args.toolchain}")
        if suite.portable is not None:
            print(f"  {suite.portable_name:<9}: library call, timed like the candidate")
        print()

        for dtype in _dtypes(args) or (torch.float32,):
            precision = str(dtype).removeprefix("torch.")
            rows = []
            for shape in shapes:
                # Time in the precision being verified. Passing --dtype and then
                # timing float32 anyway would report numbers for a different program
                # from the one that just passed.
                primary = verify.InputSpec(shape=shape, dtype=dtype)
                # Weight tensors for kernels that take them, built the same way the
                # harness builds them so the timed call and the proved call agree.
                ins = suites.make_inputs(suite, primary, device=device)
                x = ins[0]

                # Verify at this exact shape before timing it. The candidate is
                # checked against the original; each additional baseline is checked
                # against the float64 oracle, because a fast baseline that computes
                # the wrong thing would silently flatter the candidate.
                summary = verify.Harness(
                    candidate=candidate,
                    reference=refs[0][1],
                    oracle=suite.oracle,
                    extras=tuple(o.build for o in suite.extras),
                    # The precision being timed, not the whole sweep: a shape is
                    # timed only after it verified, and it has to have verified
                    # in the precision the timing will report.
                    dtypes=(dtype,),
                    name=f"{suite.name}@{shape}/{precision}",
                    distributions=(verify.Distribution.NORMAL,),
                ).run([shape])
                ok = summary.ok
                failed |= not ok

                notes = []
                try:
                    measured = []
                    for base, r in refs:
                        if base.name != "original":
                            arb = verify.arbitrate(
                                r(*ins),
                                refs[0][1](*ins),
                                suite.oracle(*(t.double() for t in ins)),
                            )
                            if arb.verdict == "worse":
                                notes.append(f"{base.name} is LESS ACCURATE than the original")
                                failed = True
                        measured.append(
                            bench.Measurement(
                                base.name,
                                bench.stat_reference(r, ins, reps=args.reps, runs=args.runs),
                                "cuda",
                            )
                        )
                    if suite.portable is not None:
                        # Same timing path as the candidate, so both carry the same
                        # host-side dispatch cost and the ratio between them is fair.
                        measured.append(
                            bench.Measurement(
                                suite.portable_name,
                                bench.stat_candidate(
                                    suite.portable, ins, reps=args.reps, runs=args.runs
                                ),
                                device,
                            )
                        )
                    cand = bench.Measurement(
                        "candidate",
                        bench.stat_candidate(candidate, ins, reps=args.reps, runs=args.runs),
                        device,
                    )
                except RuntimeError as exc:
                    print(f"  timing unavailable at {shape}: {exc}")
                    failed = True
                    continue

                rows.append(
                    bench.ShapeRow(
                        shape=tuple(x.shape),
                        candidate=cand,
                        baselines=tuple(measured),
                        verified=ok,
                        notes=notes,
                    )
                )

            table = bench.render(rows)
            print(f"  dtype: {precision}")
            print(table)
            print()
            report_sections.append(
                "\n".join(
                    [
                        f"## {suite.name} ({precision})",
                        "",
                        f"candidate: {described}",
                        f"reps: {args.reps}, runs: {args.runs}, device: `{device}`",
                        "",
                        "```",
                        table,
                        "```",
                    ]
                )
            )

        print("median of runs, [min-max] beside it. latency-bound marks shapes where")
        print("the candidate never reaches the throughput it shows at larger sizes.")

    if args.report and report_sections:
        from hipbridge.verify import provenance

        written = provenance.write(
            _report_path(args, "bench"),
            "hipbridge benchmark",
            args.toolchain,
            args.arch,
            "\n\n".join(report_sections),
        )
        print(f"\nreport written to {written}")

    return 1 if (failed and args.require) else 0


def _cmd_port(args) -> int:
    """recognize -> propose a substitute -> prove it on device -> emit the code.

    The whole pipeline in one command, and the only one of them that answers the
    question a user actually has: can I replace this kernel, and how do I know.

    Exit codes are the interface here: 0 substituted and proved, 3 recognized but
    nothing to propose, 4 proposed but the proof failed, 5 no device to prove it
    on. A substitution is never recommended on the strength of recognition alone.
    """
    kernels = parse_file(args.file)
    if not kernels:
        print(f"no kernel definitions found in {args.file}", file=sys.stderr)
        return 1

    facts = kernels[0]
    if args.kernel:
        match = [k for k in kernels if k.name == args.kernel]
        if not match:
            names = ", ".join(k.name for k in kernels)
            print(f"no kernel named {args.kernel} in {args.file} (found: {names})")
            return 1
        facts = match[0]

    result = recognize(facts)
    print(result.report())
    print()

    from hipbridge import verify

    if not verify.available():
        print("SKIP: [verify] extra not installed, so no substitution can be proved")
        return 1 if args.require else 0

    from hipbridge.verify import substitutions, suites

    source = Path(args.file).read_text(encoding="utf-8")
    notes: list[str] = []
    proposal = substitutions.propose(facts, result.pattern, notes, epsilon=args.eps)
    if proposal is None:
        print("no substitution proposed.")
        print()
        # A near miss is worth explaining. Without this, a kernel that matched
        # everything except the order of its weights reports the same "nothing
        # to propose" as a kernel nobody recognised, and sends its author
        # looking for a missing feature instead of reading their signature.
        for note in notes:
            print(f"  {note}")
        if notes:
            print()
        if result.recognized:
            print(f"  {facts.name} is a {result.pattern.value}, but a pattern is not a")
            print("  licence to substitute: several different computations share it.")
            print("  Nothing here matched this kernel's maths closely enough to try.")
        else:
            print("  no recognizer claimed this kernel, so there is nothing to propose.")
        print()
        print("  Translate it by hand, or extend hipbridge.verify.substitutions.")
        return 3

    candidate, described = suites.candidate_for(proposal.suite, proposal.epsilon)
    print(f"proposing: {described}")
    # The structural read that got us here, restated beside the evidence rather
    # than left twenty lines up. It is advisory: a "likely" match is proposed
    # exactly as readily as a "certain" one, because the proof that follows is
    # stronger evidence than the recognizer could ever be.
    print(f"  - recognized as {result.pattern.value} ({result.confidence}), which is advisory")
    for e in proposal.evidence:
        print(f"  - {e}")
    print()

    # The proof compiles the CALLER'S kernel, not the shipped example. Proving a
    # substitution against our own copy of the original would prove nothing.
    block = None
    if args.block:
        block = (int(args.block), 1, 1)
    launch = substitutions.reference_launch(proposal.suite, facts, block)
    ref = verify.NativeReference(
        source=source,
        launch=launch,
        toolchain=args.toolchain,
        command_prefix=verify.wsl(args.wsl) if args.wsl else [],
        extra_flags=(["--offload-arch=" + args.arch] if args.arch else []),
    )
    avail = ref.availability()
    if not avail:
        print(f"CANNOT PROVE: {avail.reason}")
        print()
        print("  The substitution is unproven, so it is not recommended. Re-run this")
        print("  on a machine with the toolchain and a device.")
        return 5 if args.require else 0

    print(f"proving against {args.file} compiled with {args.toolchain}")
    print(f"  launch: grid one block per row, block={launch.block}")

    summary = verify.Harness(
        candidate=candidate,
        reference=ref,
        oracle=suites.oracle_for(proposal.suite, proposal.epsilon),
        extras=tuple(o.build for o in proposal.suite.extras),
        name=f"{facts.name} vs {described}",
    ).run(list(proposal.suite.shapes())[: args.limit])
    print(summary)
    print()

    # The probe now lives in Harness, so verify, bench and synth get it too.
    # What stays here is the advice only port can give: it is the command that
    # inferred the launch geometry, so it is the one that can suggest --block.
    if summary.probe_failure:
        print(f"  Inferred block={launch.block}; override it with --block.")
        print("  No substitution is claimed, because being better than a broken")
        print("  reference is not evidence of anything.")
        return 4

    if not summary.ok:
        print("SUBSTITUTION REJECTED. The proposal did not survive verification,")
        print("which is the system working: recognition proposes, the oracle decides.")
        return 4

    print("SUBSTITUTION PROVED. Use it like this:")
    print()
    print(f"    {proposal.suite.usage_import}")
    print()
    print(f"    {proposal.suite.usage_call}   # replaces {facts.name}")
    print()
    print(
        f"Proved on {args.toolchain}"
        f"{' for ' + args.arch if args.arch else ''} against your own kernel,"
    )
    print("judged against a float64 oracle rather than against the original's rounding.")
    if args.report:
        from hipbridge.verify import provenance

        body = "\n".join(
            [
                f"source: `{args.file}`  kernel: `{facts.name}`",
                f"pattern: `{result.pattern.value}` ({result.confidence})",
                f"substitute: {described}",
                f"launch: `block={launch.block}`",
                "",
                "```",
                str(summary),
                "```",
            ]
        )
        written = provenance.write(
            _report_path(args, f"port-{facts.name}"),
            "hipbridge port report",
            args.toolchain,
            args.arch,
            body,
        )
        print(f"report written to {written}")
    return 0


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
    kernels = parse_file(args.file)
    if not kernels:
        print(f"no kernel definitions found in {args.file}", file=sys.stderr)
        return 1
    facts = kernels[0]

    from hipbridge import synth, verify

    if not verify.available():
        print("SKIP: [verify] extra not installed, so nothing could be proved")
        return 1 if args.require else 0

    from hipbridge.verify import substitutions

    source = Path(args.file).read_text(encoding="utf-8")
    result = recognize(facts)
    proposal = substitutions.propose(facts, result.pattern)
    if proposal is None:
        print("no suite covers this kernel, so there is no oracle to judge against.")
        print("Generation without a proof is not something this tool will do.")
        return 3

    suite = proposal.suite
    ref = verify.NativeReference(
        source=source,
        launch=substitutions.reference_launch(suite, facts),
        toolchain=args.toolchain,
        command_prefix=verify.wsl(args.wsl) if args.wsl else [],
        extra_flags=(["--offload-arch=" + args.arch] if args.arch else []),
    )
    avail = ref.availability()
    if not avail:
        print(f"CANNOT PROVE: {avail.reason}")
        print("Generated code is worth nothing unproved, so nothing is generated.")
        return 5 if args.require else 0

    def verify_one(fn):
        return verify.Harness(
            candidate=fn,
            reference=ref,
            oracle=suite.oracle,
            extras=tuple(o.build for o in suite.extras),
            name="generated",
        ).run(list(suite.shapes())[: args.limit])

    generator = (
        synth.ScriptGenerator(command=args.generator.split())
        if args.generator
        else synth.StaticGenerator(
            sources=[Path(p).read_text(encoding="utf-8") for p in args.file_candidate]
        )
    )

    prompt = synth.prompt_for(source, facts.name, args.arch or "gfx942")
    if args.show_prompt:
        print(prompt)
        print()

    print(f"generating up to {args.count} candidate(s) for {facts.name}")
    print(f"judged against {args.file} compiled with {args.toolchain}, and a float64 oracle")
    print()

    try:
        found = synth.search(
            generator,
            prompt,
            verify_one,
            count=args.count,
            allow_untrusted_code=args.allow_untrusted_code,
        )
    except synth.UntrustedCodeError as exc:
        print(f"REFUSED: {exc}")
        print()
        print("  Generated kernels execute in this process with its privileges.")
        print("  Pass --allow-untrusted-code once you are on a machine where that")
        print("  is acceptable, which is not one holding credentials you care about.")
        return 2

    print(found)
    winner = found.winner
    if winner is None:
        return 1 if args.require else 4

    print()
    print(winner.summary)
    if args.out:
        Path(args.out).write_text(winner.candidate.source, encoding="utf-8")
        print(f"kernel written to {args.out}")
    return 0


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
    ver.add_argument(
        "--dtype",
        action="append",
        default=[],
        choices=["float32", "float16", "bfloat16"],
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
        choices=["float32", "float16", "bfloat16"],
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
        "--require",
        action="store_true",
        help="fail instead of skipping when nothing can be proved",
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
    syn.add_argument("--require", action="store_true")
    syn.set_defaults(fn=_cmd_synth)

    info = sub.add_parser("info", help="show capabilities and which extras are installed")
    info.set_defaults(fn=_cmd_info)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())

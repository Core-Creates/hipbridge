"""The workflows, as library calls: verify, bench, port, synth.

These lived in `cli.py` as argparse callbacks, 583 of its 912 lines. Roughly 480
of those were orchestration rather than argument handling, and none of it could
be reached without going through `main([...])`. `port` in particular is the
command this project exists for, and there was no way to run it from Python.

The cost was not only ergonomic. `bench` decided a question of verification
policy inside an argparse callback:

    if arb.verdict == "worse":
        notes.append(f"{base.name} is LESS ACCURATE than the original")
        failed = True

"A tuned baseline that computes the wrong thing must not be allowed to flatter
the candidate" is a rule about what a benchmark means. It belongs where the rest
of the meaning lives, and it is now testable without spawning a CLI.

Progress is streamed through a callback rather than returned at the end, because
a full sweep runs close to 45 minutes on a metered MI300X and a caller watching
it needs to see suites finish. The CLI passes `print`; a library caller can pass
nothing and read the result object.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import torch

from hipbridge.frontend import parse_file
from hipbridge.recognizers import recognize

Progress = Callable[[str], None]


def _silent(_: str) -> None:
    """The default progress sink: a library caller reads the result instead."""


class Outcome(str, Enum):
    """What happened, in the vocabulary the exit codes are derived from.

    Deliberately not the exit codes themselves. Those are the CLI's contract
    with a shell, and a library caller should not have to know that 4 means
    rejected. What is shared is the distinction the codes were introduced for:
    a claim that failed and a claim that was never tested are both "not proved"
    and call for opposite responses.
    """

    OK = "ok"
    USAGE = "usage"
    REFUSED = "refused"
    NOTHING = "nothing"
    REJECTED = "rejected"
    UNPROVABLE = "unprovable"


# The precisions a sweep can ask for. One definition, so the CLI's --dtype
# choices, the harness and any library caller cannot disagree about the set.
DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def dtypes_by_name(names: Sequence[str] | None) -> tuple[torch.dtype, ...]:
    """Resolve --dtype names. Empty means float32 alone, the harness default."""
    out = []
    for name in names or ():
        if name not in DTYPES:
            raise ValueError(f"unknown dtype {name}; choose from {', '.join(DTYPES)}")
        out.append(DTYPES[name])
    return tuple(out)


@dataclass(frozen=True)
class Device:
    """Where and how to compile and run the reference kernel."""

    toolchain: str = "nvcc"
    arch: str = ""
    wsl: str = ""
    examples: Path = Path("examples")

    @property
    def extra_flags(self) -> list[str]:
        return ["--offload-arch=" + self.arch] if self.arch else []

    @property
    def prefix(self) -> list[str]:
        from hipbridge import verify

        return verify.wsl(self.wsl) if self.wsl else []

    def reference(self, source: str, launch):
        from hipbridge import verify

        return verify.NativeReference(
            source=source,
            launch=launch,
            toolchain=self.toolchain,
            command_prefix=self.prefix,
            extra_flags=self.extra_flags,
        )


@dataclass(frozen=True)
class Sweep:
    """What to sweep over, beyond the shapes a suite declares."""

    dtypes: tuple[torch.dtype, ...] = ()
    limit: int | None = None
    layouts: bool = False

    def shapes_of(self, suite) -> list[tuple[int, ...]]:
        shapes = list(suite.shapes())
        return shapes[: self.limit] if self.limit else shapes

    @property
    def layout_set(self):
        from hipbridge import verify

        return verify.NON_CONTIGUOUS if self.layouts else ()


# --- verify ----------------------------------------------------------------


@dataclass
class SuiteRun:
    name: str
    ok: bool
    candidate: str = ""
    skipped_reason: str | None = None
    summary: Any = None

    def as_dict(self) -> dict:
        if self.summary is not None:
            return self.summary.as_dict()
        return {"name": self.name, "ok": self.ok, "skipped_reason": self.skipped_reason}


@dataclass
class VerifyRun:
    toolchain: str
    arch: str
    suites: list[SuiteRun] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)

    @property
    def ran(self) -> int:
        return sum(1 for s in self.suites if s.summary is not None)

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.suites if s.summary is not None)

    @property
    def outcome(self) -> Outcome:
        if self.ran == 0:
            return Outcome.UNPROVABLE
        return Outcome.OK if self.ok else Outcome.REJECTED

    def as_dict(self) -> dict:
        return {
            "toolchain": self.toolchain,
            "arch": self.arch or None,
            "suites": [s.as_dict() for s in self.suites],
            "ran": self.ran,
        }

    def report_body(self) -> str:
        return "\n\n".join("```\n" + x + "\n```" for x in self.lines)


def run_suites(device: Device, sweep: Sweep, *, progress: Progress = _silent) -> VerifyRun:
    """Prove every built-in suite against a compiled reference on this machine."""
    from hipbridge import verify
    from hipbridge.verify import suites

    run = VerifyRun(toolchain=device.toolchain, arch=device.arch)

    for suite in suites.BUILTIN:
        ref = device.reference(suite.source(device.examples), suite.launch)
        avail = ref.availability()
        if not avail:
            msg = f"SKIP  {suite.name}: {avail.reason}"
            progress(msg)
            run.lines.append(msg)
            run.suites.append(SuiteRun(suite.name, ok=False, skipped_reason=avail.reason))
            continue

        candidate, described = suites.candidate_for(suite)
        progress(f"running {suite.name} against {device.toolchain} device")
        progress(f"  candidate: {described}")

        summary = verify.Harness(
            candidate=candidate,
            reference=ref,
            oracle=suite.oracle,
            extras=tuple(o.build for o in suite.extras),
            dtypes=sweep.dtypes,
            layouts=sweep.layout_set,
            name=f"{suite.name} vs original on {device.toolchain}",
        ).run(sweep.shapes_of(suite))

        progress(str(summary))
        run.lines.append(str(summary))
        run.suites.append(
            SuiteRun(suite.name, ok=bool(summary.ok), candidate=described, summary=summary)
        )

    return run


# --- bench -----------------------------------------------------------------


@dataclass
class BenchRun:
    toolchain: str
    arch: str
    device: str
    shapes: list[tuple[int, ...]]
    measurements: list[dict] = field(default_factory=list)
    sections: list[str] = field(default_factory=list)
    failed: bool = False

    @property
    def ok(self) -> bool:
        return not self.failed

    @property
    def outcome(self) -> Outcome:
        return Outcome.OK if self.ok else Outcome.REJECTED

    def as_dict(self) -> dict:
        return {
            "toolchain": self.toolchain,
            "arch": self.arch or None,
            "device": self.device,
            "shapes": [list(s) for s in self.shapes],
            "measurements": self.measurements,
        }

    def report_body(self) -> str:
        return "\n\n".join(self.sections)


def bench_suites(
    device: Device,
    sweep: Sweep,
    shapes: Sequence[tuple[int, ...]],
    *,
    reps: int,
    runs: int,
    progress: Progress = _silent,
) -> BenchRun:
    """Time every suite, and refuse to report a ratio that is not backed by a proof.

    Two rules live here rather than in the caller, because both are statements
    about what a number means:

      a shape is timed only after it verified, in the precision being timed
      a baseline less accurate than the original fails the run

    The second is the reason the second baseline exists at all. Against the
    naive original the substitutions looked like a win everywhere; against a
    competently written HIP kernel they lose below about 16M elements. A
    baseline computing the wrong thing quickly would hand back the flattering
    answer and nothing would say so.
    """
    from hipbridge import verify
    from hipbridge.verify import bench, suites

    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    run = BenchRun(
        toolchain=device.toolchain,
        arch=device.arch,
        device=torch_device,
        shapes=[tuple(s) for s in shapes],
    )

    if torch_device != "cuda":
        progress("WARNING: no GPU visible to torch, so the candidate runs on the host")
        progress("         while the original runs on device. Ratios are suppressed as")
        progress("         NOT COMPARABLE. On Windows this is normally a CPU-only torch")
        progress("         wheel; Triton has no Windows build either.")
        progress("")

    for suite in suites.BUILTIN:
        refs = []
        for base in suite.all_baselines():
            r = device.reference(base.source(device.examples), base.launch)
            avail = r.availability()
            if not avail:
                progress(f"SKIP  {suite.name} [{base.name}]: {avail.reason}")
                continue
            refs.append((base, r))
        if not refs:
            continue

        candidate, described = suites.candidate_for(suite)
        progress(
            f"{suite.name} on {device.toolchain}, device={torch_device}, reps={reps}, runs={runs}"
        )
        progress(f"  candidate: {described}")
        for base, _ in refs:
            progress(f"  {base.name:<9}: {base.source_file} compiled with {device.toolchain}")
        if suite.portable is not None:
            progress(f"  {suite.portable_name:<9}: library call, timed like the candidate")
        progress("")

        for dtype in sweep.dtypes or (torch.float32,):
            precision = str(dtype).removeprefix("torch.")
            rows = []
            for shape in shapes:
                primary = verify.InputSpec(shape=shape, dtype=dtype)
                ins = suites.make_inputs(suite, primary, device=torch_device)

                # Verified at this exact shape and precision before it is timed.
                # Timing float32 after passing --dtype float16 would report
                # numbers for a different program from the one that passed.
                summary = verify.Harness(
                    candidate=candidate,
                    reference=refs[0][1],
                    oracle=suite.oracle,
                    extras=tuple(o.build for o in suite.extras),
                    dtypes=(dtype,),
                    name=f"{suite.name}@{shape}/{precision}",
                    distributions=(verify.Distribution.NORMAL,),
                ).run([shape])
                ok = bool(summary.ok)
                run.failed |= not ok

                notes: list[str] = []
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
                                run.failed = True
                        measured.append(
                            bench.Measurement(
                                base.name,
                                bench.stat_reference(r, ins, reps=reps, runs=runs),
                                "cuda",
                            )
                        )
                    if suite.portable is not None:
                        # Same timing path as the candidate, so both carry the
                        # same host-side dispatch cost and the ratio is fair.
                        measured.append(
                            bench.Measurement(
                                suite.portable_name,
                                bench.stat_candidate(suite.portable, ins, reps=reps, runs=runs),
                                torch_device,
                            )
                        )
                    cand = bench.Measurement(
                        "candidate",
                        bench.stat_candidate(candidate, ins, reps=reps, runs=runs),
                        torch_device,
                    )
                except RuntimeError as exc:
                    progress(f"  timing unavailable at {shape}: {exc}")
                    run.failed = True
                    continue

                rows.append(
                    bench.ShapeRow(
                        shape=tuple(ins[0].shape),
                        candidate=cand,
                        baselines=tuple(measured),
                        verified=ok,
                        notes=notes,
                    )
                )

            run.measurements.append(
                {
                    "suite": suite.name,
                    "precision": precision,
                    "candidate": described,
                    "device": torch_device,
                    "reps": reps,
                    "runs": runs,
                    "rows": [r.as_dict() for r in rows],
                }
            )

            table = bench.render(rows)
            progress(f"  dtype: {precision}")
            progress(table)
            progress("")
            run.sections.append(
                "\n".join(
                    [
                        f"## {suite.name} ({precision})",
                        "",
                        f"candidate: {described}",
                        f"reps: {reps}, runs: {runs}, device: `{torch_device}`",
                        "",
                        "```",
                        table,
                        "```",
                    ]
                )
            )

        progress("median of runs, [min-max] beside it. latency-bound marks shapes where")
        progress("the candidate never reaches the throughput it shows at larger sizes.")

    return run


# --- port ------------------------------------------------------------------


@dataclass
class PortOutcome:
    """recognize -> propose -> prove, and what to tell the caller at each stop."""

    path: Path
    outcome: Outcome
    recognition: Any = None
    facts: Any = None
    proposal: Any = None
    described: str = ""
    launch: Any = None
    summary: Any = None
    declined: list[str] = field(default_factory=list)
    unprovable: str | None = None
    error: str | None = None

    @property
    def proved(self) -> bool:
        return self.outcome is Outcome.OK

    def as_dict(self) -> dict:
        out: dict = {"file": str(self.path), "proved": self.proved}
        if self.recognition is not None:
            out["recognition"] = self.recognition.as_dict()
        if self.proposal is not None:
            out["proposal"] = {
                "suite": self.proposal.suite.name,
                "substitute": self.described,
                "epsilon": self.proposal.epsilon,
                "evidence": list(self.proposal.evidence),
                "usage_import": self.proposal.suite.usage_import,
                "usage_call": self.proposal.suite.usage_call,
            }
        else:
            out["proposal"] = None
        if self.declined:
            out["declined"] = list(self.declined)
        if self.unprovable:
            out["unprovable"] = self.unprovable
        if self.summary is not None:
            out["summary"] = self.summary.as_dict()
        if self.launch is not None:
            out["launch"] = list(self.launch.block)
        return out

    def report_body(self) -> str:
        return "\n".join(
            [
                f"source: `{self.path}`  kernel: `{self.facts.name}`",
                f"pattern: `{self.recognition.pattern.value}` ({self.recognition.confidence})",
                f"substitute: {self.described}",
                f"launch: `block={self.launch.block}`",
                "",
                "```",
                str(self.summary),
                "```",
            ]
        )


def read_kernels(path: str | Path):
    """Parse a .cu file, returning (kernels, error). Never raises on a bad path."""
    try:
        kernels = parse_file(path)
    except OSError as exc:
        return None, f"cannot read {path}: {exc.strerror or exc}"
    if not kernels:
        return None, f"no kernel definitions found in {path}"
    return kernels, None


def port_file(
    path: str | Path,
    device: Device,
    sweep: Sweep,
    *,
    kernel: str | None = None,
    eps: float | None = None,
    block: int | None = None,
    progress: Progress = _silent,
) -> PortOutcome:
    """Recognize a kernel, propose a substitute, and prove it against that kernel.

    The proof compiles the CALLER'S kernel, never the shipped example. Proving a
    substitution against our own copy of the original would prove nothing.
    """
    from hipbridge import verify
    from hipbridge.verify import substitutions, suites

    path = Path(path)
    kernels, error = read_kernels(path)
    if error:
        return PortOutcome(path=path, outcome=Outcome.USAGE, error=error)

    facts = kernels[0]
    if kernel:
        match = [k for k in kernels if k.name == kernel]
        if not match:
            names = ", ".join(k.name for k in kernels)
            return PortOutcome(
                path=path,
                outcome=Outcome.USAGE,
                error=f"no kernel named {kernel} in {path} (found: {names})",
            )
        facts = match[0]

    result = recognize(facts)
    progress(result.report())
    progress("")

    notes: list[str] = []
    proposal = substitutions.propose(facts, result.pattern, notes, epsilon=eps)
    if proposal is None:
        return PortOutcome(
            path=path,
            outcome=Outcome.NOTHING,
            recognition=result,
            facts=facts,
            declined=notes,
        )

    candidate, described = suites.candidate_for(proposal.suite, proposal.epsilon)
    progress(f"proposing: {described}")
    # The structural read that got us here, restated beside the evidence rather
    # than left twenty lines up. It is advisory: a "likely" match is proposed
    # exactly as readily as a "certain" one, because the proof that follows is
    # stronger evidence than the recognizer could ever be.
    progress(f"  - recognized as {result.pattern.value} ({result.confidence}), which is advisory")
    for item in proposal.evidence:
        progress(f"  - {item}")
    progress("")

    launch = substitutions.reference_launch(proposal.suite, facts, (block, 1, 1) if block else None)
    ref = device.reference(path.read_text(encoding="utf-8"), launch)

    avail = ref.availability()
    if not avail:
        return PortOutcome(
            path=path,
            outcome=Outcome.UNPROVABLE,
            recognition=result,
            facts=facts,
            proposal=proposal,
            described=described,
            launch=launch,
            unprovable=avail.reason,
        )

    progress(f"proving against {path} compiled with {device.toolchain}")
    progress(f"  launch: grid one block per row, block={launch.block}")

    summary = verify.Harness(
        candidate=candidate,
        reference=ref,
        oracle=suites.oracle_for(proposal.suite, proposal.epsilon),
        extras=tuple(o.build for o in proposal.suite.extras),
        layouts=sweep.layout_set,
        name=f"{facts.name} vs {described}",
    ).run(sweep.shapes_of(proposal.suite))

    progress(str(summary))
    progress("")

    # A probe failure means the reference itself is broken, which makes the
    # comparison meaningless in either direction. Being better than a broken
    # reference is not evidence of anything.
    proved = bool(summary.ok) and not summary.probe_failure
    return PortOutcome(
        path=path,
        outcome=Outcome.OK if proved else Outcome.REJECTED,
        recognition=result,
        facts=facts,
        proposal=proposal,
        described=described,
        launch=launch,
        summary=summary,
    )


# --- synth -----------------------------------------------------------------


@dataclass
class SynthOutcome:
    path: Path
    outcome: Outcome
    found: Any = None
    prompt: str = ""
    error: str | None = None

    @property
    def winner(self):
        return self.found.winner if self.found is not None else None


def synth_file(
    path: str | Path,
    device: Device,
    sweep: Sweep,
    generator,
    *,
    count: int = 1,
    allow_untrusted_code: bool = False,
    show_prompt: bool = False,
    progress: Progress = _silent,
) -> SynthOutcome:
    """Ask a generator for a kernel, then let the oracle decide whether to keep it.

    Nothing is recommended on the strength of having been generated. A candidate
    that passes has matched the caller's compiled kernel across the same sweep
    every shipped kernel had to pass; one that fails is recorded and discarded.
    """
    from hipbridge import synth, verify
    from hipbridge.verify import substitutions

    path = Path(path)
    kernels, error = read_kernels(path)
    if error:
        return SynthOutcome(path=path, outcome=Outcome.USAGE, error=error)

    facts = kernels[0]
    source = path.read_text(encoding="utf-8")
    result = recognize(facts)
    proposal = substitutions.propose(facts, result.pattern)
    if proposal is None:
        return SynthOutcome(
            path=path,
            outcome=Outcome.NOTHING,
            error="no suite covers this kernel, so there is no oracle to judge against",
        )

    suite = proposal.suite
    ref = device.reference(source, substitutions.reference_launch(suite, facts))
    avail = ref.availability()
    if not avail:
        return SynthOutcome(path=path, outcome=Outcome.UNPROVABLE, error=avail.reason)

    def verify_one(fn):
        return verify.Harness(
            candidate=fn,
            reference=ref,
            oracle=suite.oracle,
            extras=tuple(o.build for o in suite.extras),
            name="generated",
        ).run(sweep.shapes_of(suite))

    prompt = synth.prompt_for(source, facts.name, device.arch or "gfx942")
    if show_prompt:
        progress(prompt)
        progress("")

    progress(f"generating up to {count} candidate(s) for {facts.name}")
    progress(f"judged against {path} compiled with {device.toolchain}, and a float64 oracle")
    progress("")

    try:
        found = synth.search(
            generator,
            prompt,
            verify_one,
            count=count,
            allow_untrusted_code=allow_untrusted_code,
        )
    except synth.UntrustedCodeError as exc:
        return SynthOutcome(path=path, outcome=Outcome.REFUSED, prompt=prompt, error=str(exc))

    outcome = Outcome.OK if found.winner is not None else Outcome.REJECTED
    return SynthOutcome(path=path, outcome=outcome, found=found, prompt=prompt)


__all__ = [
    "DTYPES",
    "BenchRun",
    "Device",
    "Outcome",
    "PortOutcome",
    "Progress",
    "SuiteRun",
    "Sweep",
    "SynthOutcome",
    "VerifyRun",
    "bench_suites",
    "dtypes_by_name",
    "port_file",
    "read_kernels",
    "run_suites",
    "synth_file",
]

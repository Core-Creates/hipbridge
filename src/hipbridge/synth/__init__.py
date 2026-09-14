"""Generate candidate kernels, then let the oracle decide which survive.

Coverage in this project grows one hand-written Triton kernel at a time, and
that is the constraint on it being useful. A model can write those far faster
than a person, and the reason that is worth trying HERE rather than anywhere
else is that the expensive part of generated code is deciding whether to trust
it, and this repository is a machine for deciding exactly that.

So the loop is deliberately unglamorous. Ask a generator for a kernel. Run it
against the caller's own `.cu` on device, judged by a float64 oracle. Keep the
first that passes; discard the rest without ceremony and without a human ever
reading them. Nothing here tries to make the model better, and nothing here
softens the gate to let more through: the gate is the entire value.

What survives is not "code an LLM wrote". It is code that matched a compiled
reference across the same sweep every shipped kernel had to pass, including the
identity, unwritten-output and nondeterminism detectors.

## Running generated code is dangerous

`load()` executes Python it was handed. A generated kernel can do anything the
process can: read your keys, post them somewhere, delete files. Nothing here
sandboxes it, and pretending otherwise with a blocklist would be worse than
saying so plainly.

Callers must opt in explicitly, and the CLI requires `--allow-untrusted-code`.
Run this in a container or on a disposable box, which for AMD work is what you
are already doing. Do not run it on a machine holding credentials you care
about, and note that a rented GPU droplet holding a broadly scoped token is
exactly the machine not to run it on.
"""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from hipbridge.analysis import wavefront_for

# What a generated module must define. Fixed rather than configurable so a
# generator has one target to hit and the loader has one thing to look for.
ENTRY_POINT = "candidate"


@dataclass(frozen=True)
class Candidate:
    """One proposed implementation, and where it came from."""

    source: str
    origin: str = "unknown"
    index: int = 0

    @property
    def digest(self) -> str:
        """Short content hash, so identical proposals are recognisable as such."""
        return hashlib.sha256(self.source.encode("utf-8")).hexdigest()[:12]

    def describe(self) -> str:
        return f"{self.origin}#{self.index} ({self.digest})"


@dataclass
class Attempt:
    """What happened to one candidate. Failures are kept, not swallowed.

    A generator that produces twenty kernels which all fail the same way is
    telling you something about the prompt, and that is only visible if the
    failures are recorded rather than counted.
    """

    candidate: Candidate
    status: str  # "passed" | "rejected" | "unloadable" | "raised"
    detail: str = ""
    summary: Any = None

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def __str__(self) -> str:
        head = f"  {self.status:<11} {self.candidate.describe()}"
        return f"{head}  {self.detail}" if self.detail else head


class Generator(Protocol):
    """Anything that can propose kernels. Deliberately tiny.

    No vendor API is imported anywhere in this package. A generator is a
    callable that yields source strings, which keeps this testable without a
    network and keeps the choice of model out of the library.
    """

    def propose(self, prompt: str, count: int) -> Iterator[Candidate]: ...


@dataclass
class StaticGenerator:
    """Fixed sources, for tests and for replaying a previous run."""

    sources: Sequence[str]
    origin: str = "static"

    def propose(self, prompt: str, count: int) -> Iterator[Candidate]:
        for i, src in enumerate(list(self.sources)[:count]):
            yield Candidate(source=src, origin=self.origin, index=i)


@dataclass
class ScriptGenerator:
    """Shell out to a command that prints a Triton kernel on stdout.

    The integration point for an actual model, without this package taking a
    dependency on one. The command receives the prompt on stdin and is run once
    per candidate, so temperature and sampling are the generator's business.

    A command that fails, times out or prints nothing yields nothing rather than
    raising: an unavailable generator should end the search, not the process.
    """

    command: Sequence[str]
    origin: str = "script"
    timeout: int = 120

    def propose(self, prompt: str, count: int) -> Iterator[Candidate]:
        for i in range(count):
            try:
                r = subprocess.run(
                    list(self.command),
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
            except (OSError, subprocess.SubprocessError):
                return
            body = _strip_fences(r.stdout)
            if body.strip():
                yield Candidate(source=body, origin=self.origin, index=i)


def _strip_fences(text: str) -> str:
    """Models wrap code in markdown fences. Take the first block if present."""
    if "```" not in text:
        return text
    parts = text.split("```")
    if len(parts) < 2:
        return text
    block = parts[1]
    first, _, rest = block.partition("\n")
    # Drop a language tag like ```python
    return rest if first.strip().isalpha() else block


class UntrustedCodeError(RuntimeError):
    """Raised when generated code is asked to run without explicit consent."""


def load(candidate: Candidate, *, allow_untrusted_code: bool = False) -> Callable[..., Any]:
    """Execute a candidate and return its entry point.

    Refuses without `allow_untrusted_code`. The flag exists to make the decision
    visible at the call site rather than buried in a config file, because the
    decision is "run code a model wrote, with my privileges".
    """
    if not allow_untrusted_code:
        raise UntrustedCodeError(
            "refusing to execute generated code without allow_untrusted_code=True; "
            "it runs with this process's privileges and is not sandboxed"
        )

    namespace: dict[str, Any] = {"__name__": "hipbridge_generated"}
    exec(compile(candidate.source, f"<generated {candidate.digest}>", "exec"), namespace)  # noqa: S102

    fn = namespace.get(ENTRY_POINT)
    if not callable(fn):
        raise AttributeError(f"generated module defines no callable `{ENTRY_POINT}`")
    return fn


@dataclass
class Search:
    """The result of one generate-and-verify loop."""

    attempts: list[Attempt] = field(default_factory=list)

    @property
    def winner(self) -> Attempt | None:
        return next((a for a in self.attempts if a.passed), None)

    @property
    def rejected(self) -> list[Attempt]:
        return [a for a in self.attempts if not a.passed]

    def __str__(self) -> str:
        lines = [str(a) for a in self.attempts]
        w = self.winner
        lines.append(
            f"kept {w.candidate.describe()} after {len(self.attempts)} attempt(s)"
            if w
            else f"nothing passed after {len(self.attempts)} attempt(s)"
        )
        return "\n".join(lines)


def search(
    generator: Generator,
    prompt: str,
    verify_one: Callable[[Callable[..., Any]], Any],
    *,
    count: int = 5,
    allow_untrusted_code: bool = False,
    stop_on_pass: bool = True,
) -> Search:
    """Propose, load, verify, keep what survives.

    `verify_one` takes a candidate callable and returns something with an `.ok`
    attribute, which in practice is a `verify.Harness` summary. The signature is
    that loose on purpose: this module knows nothing about how verification
    works, only that it decides.

    Every failure mode a generated kernel actually exhibits is recorded rather
    than raised: source that will not execute, an entry point that is missing,
    a kernel that throws when called, and one that runs and is wrong. The last
    is the interesting one, and it is the only one the gate is really for.
    """
    result = Search()

    for candidate in generator.propose(prompt, count):
        try:
            fn = load(candidate, allow_untrusted_code=allow_untrusted_code)
        except UntrustedCodeError:
            raise
        except Exception as exc:  # noqa: BLE001 - a broken proposal is data, not a crash
            result.attempts.append(Attempt(candidate, "unloadable", f"{type(exc).__name__}: {exc}"))
            continue

        try:
            summary = verify_one(fn)
        except Exception as exc:  # noqa: BLE001
            result.attempts.append(Attempt(candidate, "raised", f"{type(exc).__name__}: {exc}"))
            continue

        if getattr(summary, "ok", False):
            result.attempts.append(Attempt(candidate, "passed", "", summary))
            if stop_on_pass:
                break
        else:
            first = ""
            failures = getattr(summary, "failures", None)
            if failures:
                detail = getattr(failures[0], "failures", None)
                first = detail[0] if detail else str(failures[0])
            result.attempts.append(Attempt(candidate, "rejected", first, summary))

    return result


def prompt_for(source: str, kernel: str, arch: str = "gfx942") -> str:
    """The prompt a generator is handed. Kept here so it is reviewable.

    Says what the kernel must be called and what it receives, because a
    generator that returns something differently shaped fails at the loader for
    a reason that has nothing to do with the maths.

    The wavefront width comes from the arch rather than being written in. The
    prompt used to tell every generator "64 wide, AMD CDNA", including one asked
    to target an RDNA card, which is 32 wide. Where the width is not known the
    prompt says so, and asks for sizes that are whole multiples of either.
    """
    width = wavefront_for(arch)
    target = arch or "an AMD GPU whose offload arch was not given"
    if width is None:
        sizing = [
            f"CUDA kernel, targeting {target}. Its wavefront width is not known, so use",
            "power-of-two block sizes of at least 64, which divide by either width AMD",
            "compiles to (32 or 64).",
        ]
    else:
        sizing = [
            f"CUDA kernel, targeting {target}. Wavefronts are {width} wide, so block sizes",
            f"should be multiples of {width}.",
        ]
    return "\n".join(
        [
            "Write a Triton kernel for AMD that computes the same function as this",
            *sizing,
            "",
            f"Define a module-level function named `{ENTRY_POINT}` taking the same tensors",
            "the CUDA kernel takes, in the same order, minus the output and the scalar",
            "dimensions, and returning the output tensor.",
            "",
            "Correctness is checked against a float64 oracle across shapes that straddle",
            "the wavefront width, including 63, 64 and 65, and across adversarial input",
            "distributions. Numerical stability passes matter: a softmax without its",
            "max-subtraction is bitwise correct on normal inputs and destroyed on large",
            "ones.",
            "",
            f"The kernel to reproduce is `{kernel}`:",
            "",
            source,
        ]
    )


__all__ = [
    "ENTRY_POINT",
    "Attempt",
    "Candidate",
    "Generator",
    "ScriptGenerator",
    "Search",
    "StaticGenerator",
    "UntrustedCodeError",
    "load",
    "prompt_for",
    "search",
]

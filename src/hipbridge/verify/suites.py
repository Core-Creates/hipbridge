"""Verification suites for the shipped example kernels.

A suite ties together the four things needed to prove a substitution:

  source      the original .cu, compiled and run on device as ground truth
  launch      how to launch it (grid, block, scalar arguments)
  candidate   the implementation we propose to substitute in its place
  oracle      the same maths in float64, to arbitrate when the two disagree

The example kernels compile under hipcc unchanged because they use only
constructs HIP spells identically: __global__, threadIdx, blockIdx, __shared__,
__syncthreads, fmaxf, expf. A kernel touching cudaMalloc or cuBLAS would need
hipifying first; these do not.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch

from hipbridge.verify.inputs import InputSpec, weight_distribution
from hipbridge.verify.reference import LaunchSpec


@dataclass(frozen=True)
class Baseline:
    """A native kernel to time the candidate against.

    More than one, because a single baseline can flatter the candidate. The
    original as written answers "what does replacing this buy me"; a competently
    written version of the same maths answers "is the substitution worth making
    at all", which is the question a reviewer will actually ask.
    """

    name: str
    source_file: str
    launch: LaunchSpec

    def source(self, examples_dir: Path) -> str:
        return (examples_dir / self.source_file).read_text(encoding="utf-8")


@dataclass(frozen=True)
class Operand:
    """One extra input a kernel takes, beyond the tensor being transformed.

    Carries a name because position alone is not enough. The generated driver
    binds buffers in declaration order, so a kernel written
    `(in, beta, gamma, out, ...)` would receive gamma where it expects beta and
    produce plausible, wrong output. The oracle would fail it without ever
    saying why. Names are checked against the kernel's own parameters first.

    `aliases` exist because the same operand has several conventional
    spellings: gamma is weight or scale, beta is bias or shift.
    """

    name: str
    spec: Callable[[InputSpec], InputSpec]
    aliases: tuple[str, ...] = ()
    # Applied to the generated tensor. Exists because some operands are not
    # free-floating data: RoPE's tables are a cosine and a sine of the same
    # angles, and feeding it two independent adversarial tensors asks it to
    # apply something that is not a rotation. In float32 that merely produces
    # large numbers; in fp16 the products overflow to infinity and the kernel
    # gets blamed for the sweep's choice of inputs.
    transform: Callable[[Any], Any] | None = None

    @property
    def accepted(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)

    def matches(self, param_name: str) -> bool:
        return param_name.lower().lstrip("_") in self.accepted

    def build(self, primary: InputSpec, device: str = "cpu") -> torch.Tensor:
        """The tensor this operand contributes for one case."""
        from hipbridge.verify.inputs import generate

        t = generate(self.spec(primary), device=device)
        return self.transform(t) if self.transform is not None else t


@dataclass(frozen=True)
class Suite:
    name: str
    source_file: str
    kernel: str
    launch: LaunchSpec
    oracle: Callable[[torch.Tensor], torch.Tensor]
    shapes: Callable[[], Sequence[tuple[int, ...]]]
    baselines: tuple[Baseline, ...] = ()
    # The library call a user would reach for instead of substituting anything.
    # Timed on the same device as the candidate, because "why not just use
    # torch" is the first question anyone sensible asks about a substitution.
    portable: Callable[..., torch.Tensor] | None = None
    portable_name: str = "torch"
    # Operands beyond the tensor being transformed, derived from the primary
    # case. LayerNorm's gamma and beta live here. Empty for the kernels that
    # take one tensor, which is most of them.
    extras: tuple[Operand, ...] = ()
    # How a caller actually uses the substitute once it is proved. Held per
    # suite because `port` printed a hardcoded softmax snippet for every kernel
    # it proved, so a proved LayerNorm came with instructions to call softmax on
    # it. The proof was right and the instruction was wrong, which is the worst
    # combination this project can produce.
    usage_import: str = ""
    usage_call: str = ""
    # Does this suite's maths include an epsilon? The norms do, softmax and RoPE
    # do not. Only the suites that do require the value to be read off the
    # caller's kernel before anything may be substituted into it.
    uses_epsilon: bool = False

    def source(self, examples_dir: Path) -> str:
        return (examples_dir / self.source_file).read_text(encoding="utf-8")

    def all_baselines(self) -> tuple[Baseline, ...]:
        """The original first, then any additional baselines."""
        return (Baseline("original", self.source_file, self.launch), *self.baselines)


# The epsilon to use when nothing else says. torch's default, and what every
# kernel in examples/ spells. It is a default for running the built-in suites
# against their own sources, never a value to substitute into somebody else's
# kernel: propose() declines rather than assuming it.
DEFAULT_EPS = 1e-5


def _row_shapes() -> Sequence[tuple[int, ...]]:
    from hipbridge.verify import shapes

    return shapes.sample(shapes.row_wise())


ROW_SOFTMAX = Suite(
    name="row_softmax",
    source_file="row_softmax.cu",
    kernel="row_softmax",
    launch=LaunchSpec(
        kernel="row_softmax",
        grid=lambda s: (s[0][0], 1, 1),  # one block per row
        block=(1, 1, 1),  # serial within the row, as written
        scalar_args=lambda s: [s[0][0], s[0][1]],
        out_shape=lambda s: s[0],
    ),
    oracle=lambda t: torch.softmax(t.double(), dim=-1),
    portable=lambda t: torch.softmax(t, dim=-1),
    usage_import="from hipbridge.kernels.softmax import softmax_rowwise",
    usage_call="out = softmax_rowwise(x)",
    shapes=_row_shapes,
    baselines=(
        Baseline(
            name="tuned HIP",
            source_file="row_softmax_tuned.cu",
            launch=LaunchSpec(
                kernel="row_softmax_tuned",
                grid=lambda s: (s[0][0], 1, 1),  # one block per row
                block=(256, 1, 1),  # four wavefronts, tree-reduced in LDS
                scalar_args=lambda s: [s[0][0], s[0][1]],
                out_shape=lambda s: s[0],
            ),
        ),
    ),
)


def _norm_launch(kernel: str, block: tuple[int, int, int]) -> LaunchSpec:
    """LayerNorm and RMSNorm share softmax's launch shape: one block per row."""
    return LaunchSpec(
        kernel=kernel,
        grid=lambda s: (s[0][0], 1, 1),
        block=block,
        scalar_args=lambda s: [s[0][0], s[0][1]],
        out_shape=lambda s: s[0],
    )


def _torch_layer_norm(t: torch.Tensor, eps: float = DEFAULT_EPS) -> torch.Tensor:
    """What a user would call instead of substituting anything.

    F.layer_norm uses the biased variance and no affine terms when weight and
    bias are omitted, which is exactly the maths under test. Kept in float32:
    timing the oracle would compare a float64 implementation against float32
    kernels and report a difference that is about precision, not about speed.
    """
    return torch.nn.functional.layer_norm(t, (t.shape[-1],), eps=eps)


def _torch_rms_norm(t: torch.Tensor, eps: float = DEFAULT_EPS) -> torch.Tensor:
    fn = getattr(torch.nn.functional, "rms_norm", None)
    if fn is not None:
        return fn(t, (t.shape[-1],), eps=eps)
    # Older torch: the same maths, still float32.
    return t * torch.rsqrt((t * t).mean(dim=-1, keepdim=True) + eps)


def _layer_norm_oracle(t: torch.Tensor, eps: float = DEFAULT_EPS) -> torch.Tensor:
    """float64 LayerNorm, biased variance, no affine terms.

    Biased because the kernels divide by n, not n-1. Matching the oracle to the
    maths under test is the point; an oracle computing something adjacent would
    make every case fail for a reason that has nothing to do with the kernel.
    """
    d = t.double()
    mean = d.mean(dim=-1, keepdim=True)
    centred = d - mean
    var = (centred * centred).mean(dim=-1, keepdim=True)
    return centred * torch.rsqrt(var + eps)


def _rms_norm_oracle(t: torch.Tensor, eps: float = DEFAULT_EPS) -> torch.Tensor:
    d = t.double()
    ms = (d * d).mean(dim=-1, keepdim=True)
    return d * torch.rsqrt(ms + eps)


LAYER_NORM = Suite(
    name="layer_norm",
    uses_epsilon=True,
    source_file="layer_norm.cu",
    kernel="layer_norm",
    launch=_norm_launch("layer_norm", (1, 1, 1)),  # serial within the row
    oracle=_layer_norm_oracle,
    portable=_torch_layer_norm,
    usage_import="from hipbridge.kernels.norm import layer_norm_rowwise",
    usage_call="out = layer_norm_rowwise(x)",
    portable_name="torch",
    shapes=_row_shapes,
    baselines=(
        Baseline(
            name="tuned HIP",
            source_file="layer_norm_tuned.cu",
            launch=_norm_launch("layer_norm_tuned", (256, 1, 1)),
        ),
    ),
)

RMS_NORM = Suite(
    name="rms_norm",
    uses_epsilon=True,
    source_file="rms_norm.cu",
    kernel="rms_norm",
    launch=_norm_launch("rms_norm", (1, 1, 1)),
    oracle=_rms_norm_oracle,
    portable=_torch_rms_norm,
    usage_import="from hipbridge.kernels.norm import rms_norm_rowwise",
    usage_call="out = rms_norm_rowwise(x)",
    portable_name="torch",
    shapes=_row_shapes,
    baselines=(
        Baseline(
            name="tuned HIP",
            source_file="rms_norm_tuned.cu",
            launch=_norm_launch("rms_norm_tuned", (256, 1, 1)),
        ),
    ),
)


def _layer_norm_affine_oracle(t: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor):
    """float64 affine LayerNorm. Weights arrive as float64 too, from the harness."""
    return _layer_norm_oracle(t) * gamma.double() + beta.double()


def _torch_layer_norm_affine(
    t: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = DEFAULT_EPS
):
    return torch.nn.functional.layer_norm(t, (t.shape[-1],), weight=gamma, bias=beta, eps=eps)


def _per_column(name: str, seed_offset: int, aliases: tuple[str, ...] = ()) -> Operand:
    """A weight vector as wide as one row of the primary case.

    Derived from the primary spec so a sweep over shapes sweeps the weights with
    it, and seeded apart so gamma and beta are not the same tensor twice.
    """

    def make(spec: InputSpec) -> InputSpec:
        return InputSpec(
            shape=(spec.shape[-1],),
            dtype=spec.dtype,
            distribution=weight_distribution(spec.distribution),
            seed=spec.seed + seed_offset,
        )

    return Operand(name=name, spec=make, aliases=aliases)


LAYER_NORM_AFFINE = Suite(
    name="layer_norm_affine",
    uses_epsilon=True,
    source_file="layer_norm_affine.cu",
    kernel="layer_norm_affine",
    launch=_norm_launch("layer_norm_affine", (1, 1, 1)),
    oracle=_layer_norm_affine_oracle,
    portable=_torch_layer_norm_affine,
    usage_import="from hipbridge.kernels.norm import layer_norm_affine_rowwise",
    usage_call="out = layer_norm_affine_rowwise(x, gamma, beta)",
    shapes=_row_shapes,
    extras=(
        _per_column("gamma", 1009, ("weight", "scale", "g", "w")),
        _per_column("beta", 2003, ("bias", "shift", "b")),
    ),
    baselines=(
        Baseline(
            name="tuned HIP",
            source_file="layer_norm_affine_tuned.cu",
            launch=_norm_launch("layer_norm_affine_tuned", (256, 1, 1)),
        ),
    ),
)


def _even_row_shapes() -> Sequence[tuple[int, ...]]:
    """Row shapes with an even width.

    RoPE pairs channel 2i with 2i+1, so an odd head dimension has no pairing.
    Filtering here rather than silently truncating keeps the sweep honest: the
    kernel is not defined for those shapes, so they are not claimed as passing.
    """
    from hipbridge.verify import shapes

    return [s for s in shapes.sample(shapes.row_wise()) if s[1] % 2 == 0 and s[1] >= 2]


def _torch_rms_norm_affine(
    t: torch.Tensor, gamma: torch.Tensor, eps: float = DEFAULT_EPS
) -> torch.Tensor:
    fn = getattr(torch.nn.functional, "rms_norm", None)
    if fn is not None:
        return fn(t, (t.shape[-1],), weight=gamma, eps=eps)
    return t * torch.rsqrt((t * t).mean(dim=-1, keepdim=True) + eps) * gamma


def _rms_norm_affine_oracle(
    t: torch.Tensor, gamma: torch.Tensor, eps: float = DEFAULT_EPS
) -> torch.Tensor:
    return _rms_norm_oracle(t, eps=eps) * gamma.double()


def _rope(x, cos_tab, sin_tab):
    """Rotate interleaved channel pairs. Used for both the oracle and torch.

    Same expression either way; the oracle differs only in arriving in float64,
    which is what makes it an oracle rather than a second opinion.
    """
    x0, x1 = x[:, 0::2], x[:, 1::2]
    out = torch.empty_like(x)
    out[:, 0::2] = x0 * cos_tab - x1 * sin_tab
    out[:, 1::2] = x0 * sin_tab + x1 * cos_tab
    return out


def _rope_oracle(x, cos_tab, sin_tab):
    return _rope(x.double(), cos_tab.double(), sin_tab.double())


def _half_width(
    name: str,
    seed_offset: int,
    aliases: tuple[str, ...] = (),
    transform: Callable[[Any], Any] | None = None,
) -> Operand:
    """A per-position table holding one angle per channel pair."""

    def make(spec: InputSpec) -> InputSpec:
        rows, cols = spec.shape[0], spec.shape[1]
        return InputSpec(
            shape=(rows, cols // 2),
            dtype=spec.dtype,
            # Angles, not weights. The cosine and sine transforms make the pair
            # a rotation whatever these are, so this follows the primary case
            # and the hostile inputs keep hostile angles beside them.
            distribution=spec.distribution,
            seed=spec.seed + seed_offset,
        )

    return Operand(name=name, spec=make, aliases=aliases, transform=transform)


RMS_NORM_AFFINE = Suite(
    name="rms_norm_affine",
    uses_epsilon=True,
    source_file="rms_norm_affine.cu",
    kernel="rms_norm_affine",
    launch=_norm_launch("rms_norm_affine", (1, 1, 1)),
    oracle=_rms_norm_affine_oracle,
    portable=_torch_rms_norm_affine,
    usage_import="from hipbridge.kernels.norm import rms_norm_affine_rowwise",
    usage_call="out = rms_norm_affine_rowwise(x, gamma)",
    shapes=_row_shapes,
    extras=(_per_column("gamma", 3001, ("weight", "scale", "g", "w")),),
    baselines=(
        Baseline(
            name="tuned HIP",
            source_file="rms_norm_affine_tuned.cu",
            launch=_norm_launch("rms_norm_affine_tuned", (256, 1, 1)),
        ),
    ),
)

ROPE = Suite(
    name="rope",
    source_file="rope.cu",
    kernel="rope",
    launch=_norm_launch("rope", (1, 1, 1)),
    oracle=_rope_oracle,
    portable=_rope,
    usage_import="from hipbridge.kernels.rope import rope_rowwise",
    usage_call="out = rope_rowwise(x, cos_tab, sin_tab)",
    shapes=_even_row_shapes,
    extras=(
        _half_width("cos_tab", 4001, ("cos", "cos_cache", "freqs_cos", "c"), torch.cos),
        # Same seed offset as cos, so both are taken from one set of angles and
        # the pair is an actual rotation rather than two unrelated tables.
        _half_width("sin_tab", 4001, ("sin", "sin_cache", "freqs_sin", "s"), torch.sin),
    ),
    baselines=(
        Baseline(
            name="tuned HIP",
            source_file="rope_tuned.cu",
            launch=_norm_launch("rope_tuned", (256, 1, 1)),
        ),
    ),
)


def _rms_norm_rope_oracle(x, gamma, cos_tab, sin_tab, eps: float = DEFAULT_EPS):
    """float64 RMSNorm with a learned scale, then a rotation of channel pairs."""
    normed = _rms_norm_oracle(x, eps=eps) * gamma.double()
    return _rope(normed, cos_tab.double(), sin_tab.double())


def _two_launches(x, gamma, cos_tab, sin_tab, eps: float = DEFAULT_EPS):
    """The same maths as two separate operations, which is what a library gives you.

    This is the baseline the fused kernel has to beat, and it is deliberately
    made of the project's own kernels rather than torch. Comparing one fused
    launch against two of exactly the same launches isolates what fusion buys
    from what the kernels buy, and a torch comparison would confound the two.

    Falls back to torch when Triton is absent, so the suite still means something
    on a machine without it.
    """
    from hipbridge import kernels

    if kernels.available():
        from hipbridge.kernels.norm import rms_norm_affine_rowwise
        from hipbridge.kernels.rope import rope_rowwise

        return rope_rowwise(rms_norm_affine_rowwise(x, gamma, eps=eps), cos_tab, sin_tab)
    return _rope(_torch_rms_norm_affine(x, gamma, eps=eps), cos_tab, sin_tab)


RMS_NORM_ROPE = Suite(
    name="rms_norm_rope",
    uses_epsilon=True,
    source_file="rms_norm_rope.cu",
    kernel="rms_norm_rope",
    launch=_norm_launch("rms_norm_rope", (1, 1, 1)),
    oracle=_rms_norm_rope_oracle,
    portable=_two_launches,
    portable_name="2 launches",
    usage_import="from hipbridge.kernels.fused import rms_norm_rope_rowwise",
    usage_call="out = rms_norm_rope_rowwise(x, gamma, cos_tab, sin_tab)",
    shapes=_even_row_shapes,
    extras=(
        _per_column("gamma", 6007, ("weight", "scale", "g", "w")),
        _half_width("cos_tab", 4001, ("cos", "cos_cache", "freqs_cos", "c"), torch.cos),
        _half_width("sin_tab", 4001, ("sin", "sin_cache", "freqs_sin", "s"), torch.sin),
    ),
    baselines=(
        Baseline(
            name="tuned HIP",
            source_file="rms_norm_rope_tuned.cu",
            launch=_norm_launch("rms_norm_rope_tuned", (256, 1, 1)),
        ),
    ),
)

BUILTIN: tuple[Suite, ...] = (
    ROW_SOFTMAX,
    LAYER_NORM,
    LAYER_NORM_AFFINE,
    RMS_NORM,
    RMS_NORM_AFFINE,
    ROPE,
    RMS_NORM_ROPE,
)


def make_inputs(suite: Suite, spec: InputSpec, device: str = "cpu") -> tuple[torch.Tensor, ...]:
    """The full operand list for one case: the primary tensor, then the extras.

    One place, because there were two and they drifted. `bench` built the tuple
    itself and then handed the bare primary tensor to the torch baseline, so a
    kernel taking weights crashed with a missing-argument TypeError after the
    verification had already passed. Callers that need operands should ask for
    them here rather than assembling their own.
    """
    from hipbridge.verify.inputs import generate

    return (
        generate(spec, device=device),
        *(operand.build(spec, device=device) for operand in suite.extras),
    )


def _candidate_impl(suite: Suite) -> tuple[Callable[[torch.Tensor], torch.Tensor], str]:
    """The implementation being proposed in place of the original.

    Prefers the tuned AMD Triton kernel when the [kernels] extra is installed,
    which is the substitution this project actually exists to make. Falls back to
    a torch implementation so the suite still runs and still means something on a
    machine without Triton.
    """
    from hipbridge import kernels

    have_triton = kernels.available()

    if suite.name == "row_softmax":
        if have_triton:
            from hipbridge.kernels.softmax import softmax_rowwise

            return softmax_rowwise, "hipbridge.kernels.softmax (Triton, AMD-tuned)"
        return (lambda t: torch.softmax(t, dim=-1)), "torch.softmax (Triton unavailable)"

    if suite.name == "layer_norm":
        if have_triton:
            from hipbridge.kernels.norm import layer_norm_rowwise

            return layer_norm_rowwise, "hipbridge.kernels.norm.layer_norm (Triton, AMD-tuned)"
        return _torch_layer_norm, "torch.nn.functional.layer_norm (Triton unavailable)"

    if suite.name == "layer_norm_affine":
        if have_triton:
            from hipbridge.kernels.norm import layer_norm_affine_rowwise

            return (
                layer_norm_affine_rowwise,
                "hipbridge.kernels.norm.layer_norm_affine (Triton, AMD-tuned)",
            )
        return _torch_layer_norm_affine, "torch layer_norm affine (Triton unavailable)"

    if suite.name == "rms_norm":
        if have_triton:
            from hipbridge.kernels.norm import rms_norm_rowwise

            return rms_norm_rowwise, "hipbridge.kernels.norm.rms_norm (Triton, AMD-tuned)"
        return _torch_rms_norm, "torch rms_norm (Triton unavailable)"

    if suite.name == "rms_norm_affine":
        if have_triton:
            from hipbridge.kernels.norm import rms_norm_affine_rowwise

            return (
                rms_norm_affine_rowwise,
                "hipbridge.kernels.norm.rms_norm_affine (Triton, AMD-tuned)",
            )
        return _torch_rms_norm_affine, "torch rms_norm affine (Triton unavailable)"

    if suite.name == "rms_norm_rope":
        if have_triton:
            from hipbridge.kernels.fused import rms_norm_rope_rowwise

            return rms_norm_rope_rowwise, "hipbridge.kernels.fused.rms_norm_rope (Triton, fused)"
        return _two_launches, "torch rms_norm then rope (Triton unavailable)"

    if suite.name == "rope":
        if have_triton:
            from hipbridge.kernels.rope import rope_rowwise

            return rope_rowwise, "hipbridge.kernels.rope (Triton, AMD-tuned)"
        return _rope, "torch rope (Triton unavailable)"

    raise KeyError(suite.name)


def candidate_for(
    suite: Suite, eps: float | None = None
) -> tuple[Callable[[torch.Tensor], torch.Tensor], str]:
    """The implementation being proposed, built with the caller's epsilon.

    Every substitute in kernels/ already took an eps argument; what was missing
    was anyone passing one. Left unbound, a substitution into a kernel written
    with 1e-6 quietly computed something else, and the oracle agreed with the
    substitute because it was hardcoded to the same default the substitute used.
    """
    fn, described = _candidate_impl(suite)
    if suite.uses_epsilon and eps is not None:
        fn = partial(fn, eps=eps)
        described = f"{described}, eps={eps:g}"
    return fn, described


def oracle_for(suite: Suite, eps: float | None = None):
    """The suite's float64 oracle, built with the caller's epsilon.

    An oracle is only a truth about the program it describes. Built with this
    project's default against a kernel written with another value it stops being
    one: the caller's correct kernel becomes the outlier, and the substitution
    that happens to match the oracle is scored the more accurate side while
    changing what the caller computes.
    """
    if not suite.uses_epsilon or eps is None:
        return suite.oracle
    return partial(suite.oracle, eps=eps)


__all__ = [
    "BUILTIN",
    "LAYER_NORM",
    "LAYER_NORM_AFFINE",
    "RMS_NORM",
    "RMS_NORM_AFFINE",
    "RMS_NORM_ROPE",
    "ROPE",
    "ROW_SOFTMAX",
    "Baseline",
    "Operand",
    "Suite",
    "DEFAULT_EPS",
    "candidate_for",
    "make_inputs",
    "oracle_for",
]

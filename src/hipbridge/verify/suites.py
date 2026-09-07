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
from pathlib import Path
from typing import Any

import torch

from hipbridge.verify.inputs import InputSpec
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

    def source(self, examples_dir: Path) -> str:
        return (examples_dir / self.source_file).read_text(encoding="utf-8")

    def all_baselines(self) -> tuple[Baseline, ...]:
        """The original first, then any additional baselines."""
        return (Baseline("original", self.source_file, self.launch), *self.baselines)


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


def _torch_layer_norm(t: torch.Tensor) -> torch.Tensor:
    """What a user would call instead of substituting anything.

    F.layer_norm uses the biased variance and no affine terms when weight and
    bias are omitted, which is exactly the maths under test. Kept in float32:
    timing the oracle would compare a float64 implementation against float32
    kernels and report a difference that is about precision, not about speed.
    """
    return torch.nn.functional.layer_norm(t, (t.shape[-1],), eps=1e-5)


def _torch_rms_norm(t: torch.Tensor) -> torch.Tensor:
    fn = getattr(torch.nn.functional, "rms_norm", None)
    if fn is not None:
        return fn(t, (t.shape[-1],), eps=1e-5)
    # Older torch: the same maths, still float32.
    return t * torch.rsqrt((t * t).mean(dim=-1, keepdim=True) + 1e-5)


def _layer_norm_oracle(t: torch.Tensor) -> torch.Tensor:
    """float64 LayerNorm, biased variance, no affine terms.

    Biased because the kernels divide by n, not n-1. Matching the oracle to the
    maths under test is the point; an oracle computing something adjacent would
    make every case fail for a reason that has nothing to do with the kernel.
    """
    d = t.double()
    mean = d.mean(dim=-1, keepdim=True)
    centred = d - mean
    var = (centred * centred).mean(dim=-1, keepdim=True)
    return centred * torch.rsqrt(var + 1e-5)


def _rms_norm_oracle(t: torch.Tensor) -> torch.Tensor:
    d = t.double()
    ms = (d * d).mean(dim=-1, keepdim=True)
    return d * torch.rsqrt(ms + 1e-5)


LAYER_NORM = Suite(
    name="layer_norm",
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


def _torch_layer_norm_affine(t: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor):
    return torch.nn.functional.layer_norm(t, (t.shape[-1],), weight=gamma, bias=beta, eps=1e-5)


def _per_column(name: str, seed_offset: int, aliases: tuple[str, ...] = ()) -> Operand:
    """A weight vector as wide as one row of the primary case.

    Derived from the primary spec so a sweep over shapes sweeps the weights with
    it, and seeded apart so gamma and beta are not the same tensor twice.
    """

    def make(spec: InputSpec) -> InputSpec:
        return InputSpec(
            shape=(spec.shape[-1],),
            dtype=spec.dtype,
            distribution=spec.distribution,
            seed=spec.seed + seed_offset,
        )

    return Operand(name=name, spec=make, aliases=aliases)


LAYER_NORM_AFFINE = Suite(
    name="layer_norm_affine",
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


def _torch_rms_norm_affine(t: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    fn = getattr(torch.nn.functional, "rms_norm", None)
    if fn is not None:
        return fn(t, (t.shape[-1],), weight=gamma, eps=1e-5)
    return t * torch.rsqrt((t * t).mean(dim=-1, keepdim=True) + 1e-5) * gamma


def _rms_norm_affine_oracle(t: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    return _rms_norm_oracle(t) * gamma.double()


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
            distribution=spec.distribution,
            seed=spec.seed + seed_offset,
        )

    return Operand(name=name, spec=make, aliases=aliases, transform=transform)


RMS_NORM_AFFINE = Suite(
    name="rms_norm_affine",
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

BUILTIN: tuple[Suite, ...] = (
    ROW_SOFTMAX,
    LAYER_NORM,
    LAYER_NORM_AFFINE,
    RMS_NORM,
    RMS_NORM_AFFINE,
    ROPE,
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


def candidate_for(suite: Suite) -> tuple[Callable[[torch.Tensor], torch.Tensor], str]:
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

    if suite.name == "rope":
        if have_triton:
            from hipbridge.kernels.rope import rope_rowwise

            return rope_rowwise, "hipbridge.kernels.rope (Triton, AMD-tuned)"
        return _rope, "torch rope (Triton unavailable)"

    raise KeyError(suite.name)


__all__ = [
    "BUILTIN",
    "LAYER_NORM",
    "LAYER_NORM_AFFINE",
    "RMS_NORM",
    "RMS_NORM_AFFINE",
    "ROPE",
    "ROW_SOFTMAX",
    "Baseline",
    "Operand",
    "Suite",
    "candidate_for",
    "make_inputs",
]

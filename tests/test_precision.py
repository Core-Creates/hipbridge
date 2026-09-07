"""Half precision, which is what inference actually runs in.

Everything here was float32: the ULP metric, the noise floor, the TINY
distribution, and every shipped kernel. That made the project unable to say
anything about the workload it targets. These tests pin the parts that had to
stop assuming a 24-bit mantissa.
"""

from __future__ import annotations

import pytest

from hipbridge.verify import available

pytestmark = pytest.mark.skipif(not available(), reason="[verify] extra not installed")

HALF = ["float16", "bfloat16"]


@pytest.fixture(scope="module")
def torch_():
    import torch

    return torch


@pytest.mark.parametrize("name", [*HALF, "float32", "float64"])
def test_ulp_is_counted_in_the_precision_the_kernel_used(torch_, name):
    """One ULP is one ULP, whatever the type.

    Upcasting float16 to float32 before counting turned a one-ULP disagreement
    into roughly 8192, which reads as catastrophic and is not. The count has to
    be taken in the type the kernel actually computed in.
    """
    from hipbridge.verify.compare import ulp_diff

    dtype = getattr(torch_, name)
    a = torch_.tensor([1.0], dtype=dtype)
    b = a.clone()
    view = (
        torch_.int16
        if a.element_size() == 2
        else (torch_.int32 if a.element_size() == 4 else torch_.int64)
    )
    b.view(view)[0] += 1

    assert int(ulp_diff(a, b).max()) == 1, f"{name} should read one ULP apart"


@pytest.mark.parametrize("name", HALF)
def test_the_noise_floor_follows_the_dtype(torch_, name):
    """A float32 floor applied to fp16 is about 8000x too tight.

    Two implementations exact to the last representable half-precision bit would
    be ranked against each other on rounding neither could have avoided.
    """
    from hipbridge.verify import arbitrate

    dtype = getattr(torch_, name)
    truth = torch_tensor_truth = torch_.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch_.float64)

    # Both sides rounded to the nearest representable half value: identical, and
    # differing from truth by an amount only the storage type explains.
    rounded = truth.to(dtype)
    arb = arbitrate(rounded, rounded, torch_tensor_truth)

    assert arb.verdict == "equivalent", f"{name}: {arb}"


@pytest.mark.parametrize("name", [*HALF, "float32"])
def test_the_tiny_distribution_survives_the_dtype(torch_, name):
    """Fixed at 1e-30 it flushed to zero in fp16, testing nothing at all."""
    from hipbridge.verify.inputs import Distribution, InputSpec, generate

    dtype = getattr(torch_, name)
    t = generate(InputSpec(shape=(4, 8), dtype=dtype, distribution=Distribution.TINY))

    assert int((t != 0).sum()) == t.numel(), f"{name}: TINY underflowed to zero"
    assert float(t.abs().max()) < 0.1, "TINY should still be small"


@pytest.mark.parametrize("name", HALF)
def test_a_sweep_runs_in_half_precision(torch_, name):
    """The whole harness, end to end, in the precision inference uses."""
    from hipbridge import verify

    dtype = getattr(torch_, name)
    summary = verify.Harness(
        candidate=lambda t: torch_.softmax(t, dim=-1),
        reference=lambda t: torch_.softmax(t, dim=-1),
        oracle=lambda t: torch_.softmax(t.double(), dim=-1),
        dtypes=(dtype,),
        device="cpu",
        name=name,
    ).run([(4, 64), (1, 63)])

    assert summary.ok, str(summary)
    assert summary.results, "the sweep produced no cases"
    assert all(name in r.label for r in summary.results), "cases should name their dtype"


@pytest.mark.parametrize("name", [*HALF, "float32"])
def test_a_dropped_stability_pass_is_caught_in_every_precision(torch_, name):
    """The bug this project was built around, checked in half as well as single.

    A softmax without its max-subtraction is bitwise correct on ordinary inputs
    and destroyed on large ones. Half precision does not excuse it and must not
    hide it.
    """
    from hipbridge import verify
    from hipbridge.verify.inputs import Distribution

    dtype = getattr(torch_, name)

    def no_max_subtraction(t):
        e = torch_.exp(t)
        return e / e.sum(dim=-1, keepdim=True)

    def run(dist):
        return verify.Harness(
            candidate=no_max_subtraction,
            reference=lambda t: torch_.softmax(t, dim=-1),
            oracle=lambda t: torch_.softmax(t.double(), dim=-1),
            dtypes=(dtype,),
            distributions=(dist,),
            device="cpu",
            name=name,
        ).run([(1, 1024)])

    assert run(Distribution.NORMAL).ok, "it should look fine on ordinary inputs"
    assert not run(Distribution.LARGE).ok, f"{name}: large inputs must expose it"


@pytest.mark.parametrize(
    ("name", "toolchain", "scalar", "header"),
    [
        ("float32", "hipcc", "float", ""),
        ("float16", "hipcc", "__half", "hip_fp16.h"),
        ("bfloat16", "hipcc", "__hip_bfloat16", "hip_bf16.h"),
        ("float16", "nvcc", "__half", "cuda_fp16.h"),
        ("bfloat16", "nvcc", "__nv_bfloat16", "cuda_bf16.h"),
    ],
)
def test_the_driver_is_templated_on_the_element_type(name, toolchain, scalar, header):
    """Each vendor spells half precision differently, and both must compile."""
    from hipbridge.verify.reference import scalar_tokens

    tokens = scalar_tokens(name, toolchain)

    assert tokens["scalar"] == scalar
    assert header in tokens["scalar_header"]
    if name != "float32":
        assert tokens["from_float"].startswith("__float2"), "the sentinel needs converting"


def test_an_unsupported_element_type_says_so():
    from hipbridge.verify.reference import scalar_tokens

    with pytest.raises(TypeError, match="no driver support"):
        scalar_tokens("float8_e4m3fn", "hipcc")


@pytest.mark.parametrize("name", [*HALF, "float32"])
def test_the_generated_source_declares_the_right_scalar(torch_, name):
    """The kernel sees HB_SCALAR, so one .cu compiles for every precision."""
    from hipbridge import verify
    from hipbridge.verify.suites import ROW_SOFTMAX

    ref = verify.NativeReference(
        source="__global__ void row_softmax(const HB_SCALAR*, HB_SCALAR*, int, int) {}",
        launch=ROW_SOFTMAX.launch,
        toolchain="hipcc",
    )
    ref._n_inputs, ref._n_scalars = 1, 2
    ref._dtype_name = name

    src = ref._render_driver()
    scalar = {"float32": "float", "float16": "__half", "bfloat16": "__hip_bfloat16"}[name]

    assert f"typedef {scalar} SCALAR;" in src
    assert "#define HB_SCALAR SCALAR" in src
    assert "sizeof(SCALAR)" in src, "buffers must be sized by the element type"
    assert "sizeof(float)" not in src, "no float-sized allocation should survive"


@pytest.mark.parametrize("name", [*HALF, "float32"])
def test_tensors_survive_the_round_trip_to_bytes(torch_, name):
    """The bytes the driver reads are exactly the bytes torch wrote.

    numpy has no bfloat16, so that type crosses through an int16 view. The bits
    are the point; reinterpreting them costs nothing.
    """
    from hipbridge.verify.reference import _from_bytes, _to_bytes

    dtype = getattr(torch_, name)
    t = torch_.arange(12, dtype=torch_.float32).reshape(3, 4).to(dtype)

    back = _from_bytes(_to_bytes(t), dtype).reshape(3, 4)

    assert back.dtype is dtype
    assert torch_.equal(back, t)
    assert len(_to_bytes(t)) == t.numel() * t.element_size()

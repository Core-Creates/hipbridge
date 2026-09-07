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


def test_the_native_reference_says_it_cannot_read_half(torch_):
    """A driver reading 4-byte floats handed fp16 would compare noise.

    Two half values would be reinterpreted as one float and measured against a
    correct answer, producing a failure report that says nothing true about the
    kernel. Refusing with the reason is the only honest option until the driver
    and the example kernels are templated on the element type.
    """
    from hipbridge import verify
    from hipbridge.verify.suites import ROW_SOFTMAX

    ref = verify.NativeReference(
        source="__global__ void row_softmax(const float*, float*, int, int) {}",
        launch=ROW_SOFTMAX.launch,
        toolchain="hipcc",
    )

    with pytest.raises(TypeError, match="float32-only"):
        ref(torch_.zeros((2, 4), dtype=torch_.float16))

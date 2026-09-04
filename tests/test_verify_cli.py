"""The verify command and its suites.

The important property here is the skip path. This command is wired into a
manually-triggered CI job that usually runs on a machine with no AMD device, so
absence must exit 0 with a reason, and only --require may turn that into a
failure. If that inverts, the AMD workflow either burns money or cries wolf.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hipbridge import verify
from hipbridge.cli import main

REPO = Path(__file__).resolve().parents[1]

needs_verify = pytest.mark.skipif(not verify.available(), reason="[verify] extra not installed")


def test_verify_skips_cleanly_without_a_device(capsys):
    """No hipcc on a CI runner must be a skip, not a failure."""
    rc = main(["verify", "--toolchain", "hipcc", "--limit", "1"])
    out = capsys.readouterr().out
    if "[verify] extra not installed" in out:
        pytest.skip("extra absent; covered by the require test below")
    assert rc == 0, out
    assert "SKIP" in out and "hipcc" in out


def test_require_turns_absence_into_failure(capsys):
    rc = main(["verify", "--toolchain", "hipcc", "--limit", "1", "--require"])
    assert rc == 1, capsys.readouterr().out


@needs_verify
def test_builtin_suites_are_well_formed():
    from hipbridge.verify import suites

    assert suites.BUILTIN
    for s in suites.BUILTIN:
        assert (REPO / "examples" / s.source_file).exists(), s.source_file
        assert s.kernel in s.source(REPO / "examples")
        assert s.launch.kernel == s.kernel
        assert s.shapes()


@needs_verify
def test_candidate_resolves_and_actually_runs():
    """If a candidate is offered, it must be runnable here.

    Regression: kernels.available() used to check only that Triton imports.
    Triton installs fine on a CPU-only Linux runner and then raises
    `0 active drivers` at launch, so the Triton candidate was offered on a
    machine that could not run it. Offering an unusable implementation is worse
    than offering the fallback, because the failure surfaces at launch rather
    than at selection.
    """
    import torch

    from hipbridge.verify import suites

    fn, described = suites.candidate_for(suites.ROW_SOFTMAX)
    assert callable(fn) and described
    out = fn(torch.randn(4, 32))
    assert out.shape == (4, 32)


def test_kernels_availability_means_runnable():
    """available() must answer 'can I launch', not 'did the import succeed'."""
    from hipbridge import kernels

    if not kernels.available():
        pytest.skip("[kernels] not usable here, which is the point")
    import torch

    from hipbridge.kernels.softmax import softmax_rowwise

    x = torch.randn(2, 64, device="cuda")
    assert softmax_rowwise(x).shape == (2, 64)


@needs_verify
def test_oracle_is_float64():
    import torch

    from hipbridge.verify import suites

    out = suites.ROW_SOFTMAX.oracle(torch.randn(2, 8))
    assert out.dtype is torch.float64


# --- the workflow itself ---------------------------------------------------


def _workflow(name: str) -> dict:
    text = (REPO / ".github" / "workflows" / name).read_text(encoding="utf-8")
    return yaml.safe_load(text)


def test_amd_workflow_is_valid_yaml_and_manual_only():
    """It must never acquire a push or pull_request trigger by accident."""
    wf = _workflow("amd-verify.yml")
    # PyYAML parses the bare key `on` as the boolean True.
    triggers = wf.get("on", wf.get(True))
    assert set(triggers) == {"workflow_dispatch"}, (
        f"AMD job must stay manual, found triggers: {list(triggers)}"
    )


def test_amd_workflow_has_a_cost_ceiling():
    wf = _workflow("amd-verify.yml")
    job = wf["jobs"]["verify"]
    assert job.get("timeout-minutes"), "a metered GPU job needs a hard timeout"
    assert job["timeout-minutes"] <= 60


def test_push_ci_never_requests_a_gpu_runner():
    """The always-on matrix must stay on free hosted runners."""
    wf = _workflow("ci.yml")
    for name, job in wf["jobs"].items():
        runs_on = str(job.get("runs-on", ""))
        assert "self-hosted" not in runs_on, f"job {name} would use a paid runner"


# --- the generated device drivers -------------------------------------------


@needs_verify
@pytest.mark.parametrize("toolchain", ["nvcc", "hipcc"])
def test_generated_driver_is_well_formed(toolchain):
    """Render both drivers without needing either toolchain.

    The hipcc branch cannot be exercised on this machine, and discovering a
    template typo on a metered MI300X is an expensive way to find it. The
    rendered HIP driver has been separately confirmed to compile as C++ under
    g++ with stub HIP headers.
    """
    from hipbridge.verify import suites

    s = suites.ROW_SOFTMAX
    ref = verify.NativeReference(
        source=s.source(REPO / "examples"), launch=s.launch, toolchain=toolchain
    )
    ref._n_inputs, ref._n_scalars = 1, 2
    src = ref._render_driver()

    assert "%(" not in src, "unsubstituted template token"
    assert src.count("{") == src.count("}"), "unbalanced braces"
    assert src.count("(") == src.count(")"), "unbalanced parens"
    assert s.kernel in src, "kernel source not embedded"
    assert "-12345.0f" in src, "sentinel fill missing; unwritten output would pass"

    if toolchain == "hipcc":
        assert "#include <hip/hip_runtime.h>" in src
        assert "hipLaunchKernelGGL(row_softmax, dim3(gx,gy,gz)" in src
        assert "hipMemcpyDeviceToHost" in src
        assert "cuda" not in src.lower().replace(".cu", ""), "CUDA leaked into HIP output"
    else:
        assert "#include <cuda_runtime.h>" in src
        assert "row_softmax<<<dim3(gx,gy,gz)" in src
        assert "cudaMemcpyDeviceToHost" in src


# --- benchmarking -----------------------------------------------------------


@needs_verify
def test_cross_device_comparison_is_refused():
    """A CPU candidate against a GPU reference is not a speedup.

    Observed while validating on Windows, where torch is CPU-only: the harness
    happily printed "3.6x" for a host implementation against a device one.
    Ratios like that are how misleading benchmark tables get built, so the
    comparison is refused rather than footnoted.
    """
    from hipbridge.verify.bench import BenchResult, Comparison

    cpu = Comparison(
        candidate=BenchResult("candidate", 0.5, 10, (64, 64), "cpu"),
        reference=BenchResult("original", 2.0, 10, (64, 64), "cuda"),
        verified=True,
    )
    assert not cpu.comparable
    assert "NOT COMPARABLE" in str(cpu)
    assert "4.0x" not in str(cpu), "a speedup must not be printed for mixed devices"

    gpu = Comparison(
        candidate=BenchResult("candidate", 0.5, 10, (64, 64), "cuda"),
        reference=BenchResult("original", 2.0, 10, (64, 64), "cuda"),
        verified=True,
    )
    assert gpu.comparable
    assert abs(gpu.speedup - 4.0) < 1e-9
    assert "4.0x" in str(gpu)


@needs_verify
def test_unverified_shape_is_flagged_in_the_timing_line():
    """A fast wrong kernel is not a result."""
    from hipbridge.verify.bench import BenchResult, Comparison

    c = Comparison(
        candidate=BenchResult("candidate", 0.5, 10, (64, 64), "cuda"),
        reference=BenchResult("original", 2.0, 10, (64, 64), "cuda"),
        verified=False,
    )
    assert "UNVERIFIED" in str(c)


@needs_verify
@pytest.mark.parametrize("toolchain", ["nvcc", "hipcc"])
def test_driver_emits_timing_instrumentation(toolchain):
    from hipbridge.verify import suites

    s = suites.ROW_SOFTMAX
    ref = verify.NativeReference(
        source=s.source(REPO / "examples"), launch=s.launch, toolchain=toolchain
    )
    ref._n_inputs, ref._n_scalars = 1, 2
    src = ref._render_driver()
    assert "HIPBRIDGE_REPS" in src, "timing must be opt-in via the environment"
    # The escape must reach the C source as two characters. A real newline
    # here breaks the string literal and the build, which it once did.
    assert chr(92) + "n" in src.split("KERNEL_MS")[1][:20], (
        "timing format must keep an escaped newline, not a literal one"
    )
    assert "%(" not in src
    assert src.count("{") == src.count("}")
    ev = "hipEvent_t" if toolchain == "hipcc" else "cudaEvent_t"
    assert ev in src, "must time with device events, not wall clock"

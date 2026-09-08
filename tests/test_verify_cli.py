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

from hipbridge import cli, verify
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
    # 5, not 1: the claim could not be tested, which is a different thing from
    # bad input and from a claim that was tested and failed. See the exit code
    # contract at the top of cli.py.
    assert rc == cli.EXIT_UNPROVABLE, capsys.readouterr().out


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


def _row(shape, cand, original, tuned=None, device="cuda", verified=True):
    """A ShapeRow built from plain numbers, so the reporting is testable on CPU."""
    from hipbridge.verify.bench import Measurement, ShapeRow, Stat

    bases = [Measurement("original", Stat((original,)), "cuda")]
    if tuned is not None:
        bases.append(Measurement("tuned HIP", Stat((tuned,)), "cuda"))
    return ShapeRow(
        shape=shape,
        candidate=Measurement("candidate", Stat((cand,)), device),
        baselines=tuple(bases),
        verified=verified,
    )


@needs_verify
def test_cross_device_comparison_is_refused():
    """A CPU candidate against a GPU reference is not a speedup.

    Observed while validating on Windows, where torch is CPU-only: the harness
    happily printed "3.6x" for a host implementation against a device one.
    Ratios like that are how misleading benchmark tables get built, so the
    comparison is refused rather than footnoted.
    """
    from hipbridge.verify.bench import render

    cpu = _row((64, 64), cand=0.5, original=2.0, device="cpu")
    assert not cpu.comparable
    out = render([cpu])
    assert "NOT COMPARABLE" in out
    assert "4.0x" not in out, "a speedup must not be printed for mixed devices"

    gpu = _row((64, 64), cand=0.5, original=2.0)
    assert gpu.comparable
    assert abs(gpu.speedup("original") - 4.0) < 1e-9
    assert "4.0x" in render([gpu])


@needs_verify
def test_unverified_shape_is_flagged_in_the_timing_line():
    """A fast wrong kernel is not a result."""
    from hipbridge.verify.bench import render

    assert "UNVERIFIED" in render([_row((64, 64), cand=0.5, original=2.0, verified=False)])


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


@needs_verify
def test_table_reports_a_distribution_not_a_point():
    """A single mean hides how far a small-shape timing moves between runs."""
    from hipbridge.verify.bench import Measurement, ShapeRow, Stat, render

    noisy = Stat((0.020, 0.016, 0.018, 0.019, 0.017))
    assert abs(noisy.median - 0.018) < 1e-9
    assert abs(noisy.spread - 1.25) < 1e-9
    assert "18.0 [16.0-20.0]" == noisy.us()

    row = ShapeRow(
        shape=(1, 1024),
        candidate=Measurement("candidate", noisy, "cuda"),
        baselines=(Measurement("original", Stat((0.325, 0.334)), "cuda"),),
        verified=True,
    )
    out = render([row])
    assert "[16.0-20.0]" in out, "the range has to survive into the table"


@needs_verify
def test_latency_bound_shapes_are_labelled():
    """A shape whose time does not scale with work is measuring dispatch."""
    from hipbridge.verify.bench import render

    # 1x1024 and 4096x4096 take almost the same time: the small one never gets
    # near the throughput the large one shows, so it is launch-bound.
    small = _row((1, 1024), cand=0.018, original=0.325)
    large = _row((4096, 4096), cand=0.032, original=2.430)
    out = render([small, large])

    assert small.latency_bound, "a flat time across 16000x the work is dispatch overhead"
    assert not large.latency_bound
    lines = [ln for ln in out.splitlines() if ln.strip().startswith(("1x1024", "4096x4096"))]
    assert "latency-bound" in lines[0]
    assert "latency-bound" not in lines[1]


@needs_verify
def test_a_second_baseline_gets_its_own_ratio():
    """The naive original and a competent kernel are different questions."""
    from hipbridge.verify.bench import render

    row = _row((4096, 4096), cand=0.032, original=2.430, tuned=0.040)
    assert abs(row.speedup("original") - 75.9) < 0.1
    assert abs(row.speedup("tuned HIP") - 1.25) < 0.01

    out = render([row])
    assert "vs original" in out and "vs tuned HIP" in out
    assert "75.9x" in out and "1.2x" in out


@needs_verify
def test_row_softmax_carries_a_competent_baseline():
    """The suite must ship the fair comparison, not just the flattering one."""
    from hipbridge.verify import suites

    names = [b.name for b in suites.ROW_SOFTMAX.all_baselines()]
    assert names[0] == "original"
    assert "tuned HIP" in names

    tuned = suites.ROW_SOFTMAX.baselines[0]
    src = tuned.source(REPO / "examples")
    assert tuned.launch.block == (256, 1, 1), "a fair baseline uses the whole wavefront"
    assert "__syncthreads" in src and "__shared__" in src, "expected a tree reduction"
    assert "blockIdx.x" in src


@needs_verify
def test_the_latency_bound_threshold_flips_exactly_once():
    """The 25% line was never exercised at its boundary.

    Two shapes with the same time: the larger does more work per millisecond, so
    it sets the best throughput and the smaller is judged against it. Sizing the
    small one either side of a quarter of that throughput moves the label and
    nothing else.
    """
    from hipbridge.verify.bench import LATENCY_BOUND_FRACTION, mark_latency_bound

    def sweep(small_elements):
        # Equal times, so throughput is decided purely by element count.
        small = _row((1, small_elements), cand=0.010, original=1.0)
        large = _row((1, 1000), cand=0.010, original=1.0)
        mark_latency_bound([small, large])
        return small.latency_bound, large.latency_bound

    just_under = int(1000 * LATENCY_BOUND_FRACTION) - 1  # 249 elements
    just_over = int(1000 * LATENCY_BOUND_FRACTION) + 1  # 251 elements

    assert sweep(just_under) == (True, False), "below the line must be labelled"
    assert sweep(just_over) == (False, False), "above the line must not be"


@needs_verify
def test_a_single_shape_cannot_be_judged():
    """One shape is its own best throughput, so calling it compute-bound is circular."""
    from hipbridge.verify.bench import mark_latency_bound, render

    only = _row((1, 1024), cand=0.018, original=0.325)
    judged = mark_latency_bound([only])

    assert judged is False
    assert only.latency_bound is False
    out = render([only])
    assert "latency-bound" not in out.replace("labelled latency-bound", "")
    assert "no faster run of the same kernel" in out, "silence here would be misleading"


@needs_verify
def test_the_amd_workflow_leaves_evidence_behind():
    """A run nobody watched still has to produce something attributable.

    Every hardware number in this repository was pasted in by hand from an SSH
    session, and the box producing them went down mid-measurement more than
    once. A workflow that only reports an exit code repeats that: it tells you
    something passed and nothing about what.
    """
    wf = _workflow("amd-verify.yml")
    steps = wf["jobs"]["verify"]["steps"]
    names = [str(s.get("name", "")) for s in steps]
    body = " ".join(str(s.get("run", "")) for s in steps)

    assert any("upload-artifact" in str(s.get("uses", "")) for s in steps), (
        "results must leave the runner"
    )
    upload = next(s for s in steps if "upload-artifact" in str(s.get("uses", "")))
    assert upload.get("if") == "always()", "a failed run's evidence is the useful kind"
    assert "results/" in str(upload["with"]["path"])

    assert "hipbridge.cli verify" in body, "the workflow must actually verify"
    assert "hipbridge.cli bench" in body, "and time what it verified"
    assert "hipbridge.cli port" in body, "and prove each kernel end to end"
    assert "--report" in body, "reports carry the commit, host and toolchain"
    assert names, "steps should be named so a failed run is readable"


@needs_verify
def test_the_amd_workflow_refuses_to_measure_the_torch_fallback():
    """A hardware run that measures the fallback is worse than no hardware run.

    The first CI regeneration installed a fresh CPU-only torch over the box's
    ROCm build, so `[kernels] extra: not installed` and every suite quietly used
    torch instead of the Triton kernel under test. It reported 126/126 across
    six suites and the numbers looked entirely plausible. Nothing in the output
    said the substitution had never been attempted.
    """
    wf = _workflow("amd-verify.yml")
    steps = wf["jobs"]["verify"]["steps"]

    guard = next((s for s in steps if "fallback" in str(s.get("name", "")).lower()), None)
    assert guard is not None, "a hardware run must check the tuned kernels are present"
    assert "kernels.available()" in str(guard["run"])
    assert "ubuntu-latest" in str(guard.get("if")), "the dry run has no kernels and should skip it"

    # And it must not shadow the box's ROCm interpreter to begin with.
    setup = next(s for s in steps if "setup-python" in str(s.get("uses", "")))
    assert "ubuntu-latest" in str(setup.get("if")), (
        "installing a fresh Python on the AMD box is what caused the fallback"
    )


@needs_verify
def test_the_amd_workflow_can_sweep_half_precision():
    """Inference runs in half, so the hardware run has to cover it."""
    wf = _workflow("amd-verify.yml")
    triggers = wf.get("on", wf.get(True))
    inputs = triggers["workflow_dispatch"]["inputs"]

    assert "dtypes" in inputs, "the run should choose its precisions"
    assert "float16" in str(inputs["dtypes"]["default"])
    assert "bfloat16" in str(inputs["dtypes"]["default"])


@needs_verify
def test_the_amd_workflow_still_has_no_schedule():
    """A weekly run on a metered device bills whether anything changed or not.

    Worth an explicit test rather than trusting the trigger check alone: the
    obvious fix for "nothing runs automatically" is a cron, and the obvious fix
    is the expensive one. Adding it should be a deliberate decision.
    """
    wf = _workflow("amd-verify.yml")
    triggers = wf.get("on", wf.get(True))

    assert "schedule" not in triggers
    assert set(triggers) == {"workflow_dispatch"}

    # Parsed triggers, not a text search. The first version of this test grepped
    # the file and failed on the comment explaining why there is no schedule,
    # which is the same mistake the substitution evidence made when a comment
    # mentioning "mean" caused a refusal. Read the structure, not the prose.

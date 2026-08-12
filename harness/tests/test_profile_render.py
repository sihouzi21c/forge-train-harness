"""Unit tests for harness.tools.profile_render — nsys-backed renderer.

The renderer shells out to ``nsys stats`` in production; tests inject a
fake ``_nsys_runner`` callable returning canned CSV payloads so they run
on any host (no nsys binary, no GPU).
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.profile_render import render

# ---------------------------------------------------------------------
# Canned CSV fixtures (shape mirrors real ``nsys stats --format csv``)
# ---------------------------------------------------------------------

_KERN_CSV = """Generating CUDA Kernel Statistics...
SKIPPED: foo
Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
50.0,"5,000,000",10,"500,000","500,000","500,000","500,000",0,kernel_a
30.0,"3,000,000",20,"150,000","150,000","150,000","150,000",0,kernel_b
20.0,"2,000,000",5,"400,000","400,000","400,000","400,000",0,kernel_c
"""

_MEMOP_CSV = """Generating CUDA GPU Memory Operation Statistics...
Time (%),Total Time (ns),Count,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Operation
60.0,"600,000",10,"60,000","60,000","60,000","60,000",0,[CUDA memcpy HtoD]
40.0,"400,000",5,"80,000","80,000","80,000","80,000",0,[CUDA memcpy DtoH]
"""

_API_CSV = """Generating CUDA API Statistics...
Time (%),Total Time (ns),Num Calls,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
70.0,"7,000,000",100,"70,000","70,000","70,000","70,000",0,cudaLaunchKernel
30.0,"3,000,000",50,"60,000","60,000","60,000","60,000",0,cudaMemcpyAsync
"""

_OSRT_CSV = """Generating OS Runtime Statistics...
Time (%),Total Time (ns),Num Calls,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
80.0,"8,000,000",40,"200,000","200,000","200,000","200,000",0,pthread_cond_wait
20.0,"2,000,000",10,"200,000","200,000","200,000","200,000",0,poll
"""


def _fake_runner(payloads: dict[str, str]):
    def runner(report: str, nsys_rep: Path) -> str:
        return payloads.get(report, "")

    return runner


def _default_runner():
    return _fake_runner(
        {
            "cuda_gpu_kern_sum": _KERN_CSV,
            "cuda_gpu_mem_time_sum": _MEMOP_CSV,
            "cuda_api_sum": _API_CSV,
            "osrt_sum": _OSRT_CSV,
        }
    )


def _meta(suite: str = "profile-snapshot") -> dict:
    return {
        "suite": suite,
        "world_size": 2,
        "micro_batch_size": 4,
        "seq_length": 4096,
        "grad_accum_steps": 1,
    }


def _make_nsys_rep(tmp: Path, name: str = "rank0.nsys-rep") -> Path:
    p = tmp / name
    p.write_bytes(b"")  # existence only; runner is faked
    return p


# ---------------------------------------------------------------------
# Parsing + outputs
# ---------------------------------------------------------------------


class TestKernelParsing(unittest.TestCase):
    def test_kernels_sorted_by_total_ms_and_capped(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            rep = _make_nsys_rep(tmp)
            out = tmp / "round0"
            render(
                rep,
                out,
                suite="profile-snapshot",
                run_meta=_meta(),
                profiled_steps=5,
                _nsys_runner=_default_runner(),
            )
            data = json.loads((out / "profile.json").read_text())
        kernels = data["top_kernels"]
        # 3 rows, sorted descending by per_step_ms (1ms, 0.6ms, 0.4ms at 5 steps)
        self.assertEqual([k["name"] for k in kernels], ["kernel_a", "kernel_b", "kernel_c"])
        self.assertAlmostEqual(kernels[0]["per_step_ms"], 1.0, places=3)
        self.assertAlmostEqual(kernels[1]["per_step_ms"], 0.6, places=3)
        self.assertEqual(kernels[0]["instances"], 10)
        self.assertAlmostEqual(kernels[0]["pct"], 50.0, places=2)
        # gpu_kernel_per_step_ms is the sum (10ms) over profiled_steps (5)
        self.assertAlmostEqual(data["gpu_kernel_per_step_ms"], 2.0, places=3)
        self.assertEqual(data["profiled_steps"], 5)


class TestGpuIdle(unittest.TestCase):
    """``gpu_idle = step_time - gpu_kernel`` under the single-stream
    assumption. Verified end-to-end via render() so the header line
    and JSON field stay in sync."""

    def test_gpu_idle_equals_step_minus_kernel(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            rep = _make_nsys_rep(tmp)
            out = tmp / "round0"
            # _default_runner produces 10 ms total kernel time over 5 steps
            # → gpu_kernel_per_step_ms = 2.0. Pass step_time_ms = 5.0
            # so the expected idle is 3.0 ms / step.
            render(
                rep,
                out,
                suite="profile-snapshot",
                run_meta=_meta(),
                step_time_ms=5.0,
                profiled_steps=5,
                _nsys_runner=_default_runner(),
            )
            data = json.loads((out / "profile.json").read_text())
            summary = (out / "summary.md").read_text()
        self.assertAlmostEqual(data["gpu_idle_per_step_ms"], 3.0, places=3)
        self.assertIn("gpu_idle_per_step_ms=3.000", summary)
        # The single-stream caveat must accompany the field so a reader
        # doesn't treat the derivation as multi-stream-correct.
        self.assertIn("share one stream", summary)
        self.assertIn("cuda_api / os_runtime are CPU time", summary)

    def test_gpu_idle_clamped_at_zero_when_step_less_than_kernel(self) -> None:
        """Multi-stream workload may report kernel sum > step_time
        (concurrent kernels on separate streams accumulate). Negative
        idle is meaningless to downstream readers — clamp to zero and
        rely on the prose caveat to flag the case to the agent."""
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            rep = _make_nsys_rep(tmp)
            out = tmp / "round0"
            render(
                rep,
                out,
                suite="profile-snapshot",
                run_meta=_meta(),
                step_time_ms=1.0,  # < kernel sum (2.0)
                profiled_steps=5,
                _nsys_runner=_default_runner(),
            )
            data = json.loads((out / "profile.json").read_text())
        self.assertEqual(data["gpu_idle_per_step_ms"], 0.0)


class TestMemopParsing(unittest.TestCase):
    def test_memops_sorted_and_totals(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            rep = _make_nsys_rep(tmp)
            out = tmp / "round0"
            render(
                rep,
                out,
                suite="profile-snapshot",
                run_meta=_meta(),
                profiled_steps=5,
                _nsys_runner=_default_runner(),
            )
            data = json.loads((out / "profile.json").read_text())
        ops = data["memops"]
        self.assertEqual(ops[0]["operation"], "[CUDA memcpy HtoD]")
        # 0.6ms total over 5 steps = 0.12ms per step
        self.assertAlmostEqual(ops[0]["per_step_ms"], 0.12, places=3)
        self.assertEqual(ops[0]["count"], 10)
        self.assertAlmostEqual(data["gpu_memop_per_step_ms"], 0.2, places=3)


class TestApiParsing(unittest.TestCase):
    def test_api_sorted_and_totals(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            rep = _make_nsys_rep(tmp)
            out = tmp / "round0"
            render(
                rep,
                out,
                suite="profile-snapshot",
                run_meta=_meta(),
                profiled_steps=5,
                _nsys_runner=_default_runner(),
            )
            data = json.loads((out / "profile.json").read_text())
        api = data["top_api"]
        self.assertEqual(api[0]["name"], "cudaLaunchKernel")
        # 7ms total over 5 steps = 1.4ms per step
        self.assertAlmostEqual(api[0]["per_step_ms"], 1.4, places=3)
        self.assertEqual(api[0]["calls"], 100)
        self.assertAlmostEqual(data["cuda_api_per_step_ms"], 2.0, places=3)


class TestOsrtParsing(unittest.TestCase):
    def test_osrt_table_rendered_and_normalized(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            rep = _make_nsys_rep(tmp)
            out = tmp / "round0"
            render(
                rep,
                out,
                suite="profile-snapshot",
                run_meta=_meta(),
                profiled_steps=5,
                _nsys_runner=_default_runner(),
            )
            data = json.loads((out / "profile.json").read_text())
            summary = (out / "summary.md").read_text()
        osrt = data["top_osrt"]
        self.assertEqual(osrt[0]["name"], "pthread_cond_wait")
        # 8ms total over 5 steps = 1.6ms per step
        self.assertAlmostEqual(osrt[0]["per_step_ms"], 1.6, places=3)
        self.assertEqual(osrt[0]["calls"], 40)
        # os_runtime_per_step_ms is the sum (10ms) over 5 steps
        self.assertAlmostEqual(data["os_runtime_per_step_ms"], 2.0, places=3)
        self.assertIn("OS runtime calls", summary)
        self.assertIn("pthread_cond_wait", summary)


class TestZeroSteps(unittest.TestCase):
    def test_profiled_steps_zero_no_zero_division(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            rep = _make_nsys_rep(tmp)
            out = tmp / "round0"
            render(
                rep,
                out,
                suite="profile-snapshot",
                run_meta=_meta(),
                profiled_steps=0,
                _nsys_runner=_default_runner(),
            )
            data = json.loads((out / "profile.json").read_text())
        self.assertEqual(data["profiled_steps"], 0)
        self.assertEqual(data["gpu_kernel_per_step_ms"], 0.0)
        self.assertEqual(data["gpu_memop_per_step_ms"], 0.0)
        self.assertEqual(data["cuda_api_per_step_ms"], 0.0)
        self.assertEqual(data["os_runtime_per_step_ms"], 0.0)
        # per-row per_step_ms also 0 (no samples to normalize over)
        self.assertEqual(data["top_kernels"][0]["per_step_ms"], 0.0)


# ---------------------------------------------------------------------
# Δ semantics
# ---------------------------------------------------------------------


class TestDeltaBlock(unittest.TestCase):
    def _round(
        self,
        tmp: Path,
        name: str,
        meta: dict,
        *,
        step_time_ms: float = 100.0,
        mfu: float = 0.20,
        profiled_steps: int = 5,
    ) -> Path:
        rep = _make_nsys_rep(tmp, name=f"{name}.nsys-rep")
        out = tmp / name
        render(
            rep,
            out,
            suite=str(meta["suite"]),
            run_meta=meta,
            step_time_ms=step_time_ms,
            mfu_e2e_standard=mfu,
            profiled_steps=profiled_steps,
            _nsys_runner=_default_runner(),
        )
        return out

    def test_delta_present_when_shapes_match(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            r0 = self._round(tmp, "M6_round0", _meta(), step_time_ms=100.0, mfu=0.20)
            rep = _make_nsys_rep(tmp, name="M6_round1.nsys-rep")
            r1 = tmp / "M6_round1"
            render(
                rep,
                r1,
                suite="profile-snapshot",
                run_meta=_meta(),
                step_time_ms=90.0,
                mfu_e2e_standard=0.22,
                profiled_steps=5,
                prev_dir=r0,
                _nsys_runner=_default_runner(),
            )
            summary = (r1 / "summary.md").read_text()
        self.assertIn("Δ from M6_round0", summary)
        self.assertIn("Δ step_time_ms=-10", summary)
        self.assertIn("Δ mfu_e2e_standard=+0.02", summary)
        self.assertIn("Δ gpu_kernel_per_step_ms=", summary)
        self.assertIn("Δ os_runtime_per_step_ms=", summary)
        # All tracked kernels appear since CSVs match
        self.assertIn("tracked", summary)

    def test_delta_refused_on_shape_mismatch(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            r0 = self._round(tmp, "M6_round0", _meta())
            mismatched = _meta()
            mismatched["micro_batch_size"] = 10
            rep = _make_nsys_rep(tmp, name="M6_round1.nsys-rep")
            r1 = tmp / "M6_round1"
            render(
                rep,
                r1,
                suite="profile-snapshot",
                run_meta=mismatched,
                prev_dir=r0,
                _nsys_runner=_default_runner(),
            )
            summary = (r1 / "summary.md").read_text()
        self.assertIn("Δ from previous: unavailable", summary)
        self.assertIn("shape mismatch", summary)
        self.assertIn("micro_batch_size", summary)

    def test_delta_unavailable_when_no_prev(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            rep = _make_nsys_rep(tmp)
            out = tmp / "round0"
            render(
                rep,
                out,
                suite="profile-snapshot",
                run_meta=_meta(),
                _nsys_runner=_default_runner(),
            )
            summary = (out / "summary.md").read_text()
        self.assertIn("Δ from previous: unavailable (no prior snapshot", summary)

    def test_delta_unavailable_when_prev_json_missing(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            empty_prev = tmp / "empty_prev"
            empty_prev.mkdir()
            rep = _make_nsys_rep(tmp)
            out = tmp / "round0"
            render(
                rep,
                out,
                suite="profile-snapshot",
                run_meta=_meta(),
                prev_dir=empty_prev,
                _nsys_runner=_default_runner(),
            )
            summary = (out / "summary.md").read_text()
        self.assertIn("Δ from previous: unavailable", summary)
        self.assertIn("no profile.json", summary)

    def test_delta_marks_new_kernel(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            # round 0: only kernel_a present
            prev_csv = """Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
100.0,"5,000,000",10,"500,000","500,000","500,000","500,000",0,kernel_a
"""
            rep0 = _make_nsys_rep(tmp, name="round0.nsys-rep")
            r0 = tmp / "round0"
            render(
                rep0,
                r0,
                suite="profile-snapshot",
                run_meta=_meta(),
                _nsys_runner=_fake_runner(
                    {
                        "cuda_gpu_kern_sum": prev_csv,
                        "cuda_gpu_mem_time_sum": _MEMOP_CSV,
                        "cuda_api_sum": _API_CSV,
                    }
                ),
            )
            # round 1: full default fixture (kernel_a, kernel_b, kernel_c)
            rep1 = _make_nsys_rep(tmp, name="round1.nsys-rep")
            r1 = tmp / "round1"
            render(
                rep1,
                r1,
                suite="profile-snapshot",
                run_meta=_meta(),
                prev_dir=r0,
                _nsys_runner=_default_runner(),
            )
            summary = (r1 / "summary.md").read_text()
        self.assertIn("kernel_b", summary)
        self.assertIn("| new |", summary)


# ---------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------


class TestOutputFiles(unittest.TestCase):
    def test_both_files_written_with_expected_shape(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            rep = _make_nsys_rep(tmp)
            out = tmp / "round0"
            render(
                rep,
                out,
                suite="profile-snapshot",
                run_meta=_meta(),
                step_time_ms=95.5,
                mfu_e2e_standard=0.21,
                profiled_steps=5,
                _nsys_runner=_default_runner(),
            )
            self.assertTrue((out / "summary.md").exists())
            self.assertTrue((out / "profile.json").exists())
            data = json.loads((out / "profile.json").read_text())
            self.assertEqual(data["run_meta"]["suite"], "profile-snapshot")
            self.assertEqual(data["step_time_ms"], 95.5)
            self.assertEqual(data["mfu_e2e_standard"], 0.21)
            self.assertEqual(data["run_meta"]["nsys_rep_path"], str(rep))
            summary = (out / "summary.md").read_text()
        self.assertIn("profile snapshot", summary)
        self.assertIn("profiled_steps=5", summary)
        self.assertIn("top 15 GPU kernels", summary)
        self.assertIn("GPU memory operations", summary)
        self.assertIn("top 10 CUDA API calls", summary)
        self.assertIn("OS runtime calls", summary)
        self.assertIn("kernel_a", summary)
        self.assertIn("cudaLaunchKernel", summary)

    def test_missing_nsys_rep_raises(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with self.assertRaises(FileNotFoundError):
                render(
                    tmp / "does_not_exist.nsys-rep",
                    tmp / "round0",
                    suite="profile-snapshot",
                    run_meta=_meta(),
                    _nsys_runner=_default_runner(),
                )


if __name__ == "__main__":
    unittest.main()

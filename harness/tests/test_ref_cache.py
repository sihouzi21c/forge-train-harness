"""Tests for ref-trajectory caching in :func:`evals._common.run_via_ref_script`."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.ref_script_runner import RefRun


def _make_ref_run(
    dump_dir: Path,
    *,
    loss_file: Path | None = None,
    metadata: dict | None = None,
    elapsed_s: float = 42.0,
    returncode: int = 0,
) -> RefRun:
    stdout_path = dump_dir / "ref_stdout.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text("[LOSS] step=101 global_loss=5.0\n", encoding="utf-8")
    return RefRun(
        returncode=returncode,
        stdout_path=stdout_path,
        dump_dir=dump_dir,
        loss_file=loss_file,
        elapsed_s=elapsed_s,
        timed_out=False,
        metadata=metadata or {"gate_window_start": 101, "gate_window_end": 201, "num_steps": 200},
    )


class TestRefCacheKeyDeterminism(unittest.TestCase):
    """Cache key must be stable across calls with identical inputs."""

    def test_same_inputs_produce_same_key(self) -> None:
        from evals._common import _ref_cache_key

        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "ref.sh"
            script.write_text("#!/bin/bash\necho hello\n", encoding="utf-8")
            env = {"WORLD_SIZE": "8", "NUM_STEPS_OVERRIDE": "200"}
            k1 = _ref_cache_key("long-train", script, env, repo_root=Path(tmp))
            k2 = _ref_cache_key("long-train", script, env, repo_root=Path(tmp))
            self.assertEqual(k1, k2)

    def test_different_suite_key_different_key(self) -> None:
        from evals._common import _ref_cache_key

        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "ref.sh"
            script.write_text("#!/bin/bash\necho hello\n", encoding="utf-8")
            env = {"WORLD_SIZE": "8"}
            k1 = _ref_cache_key("long-train", script, env, repo_root=Path(tmp))
            k2 = _ref_cache_key("loss-gate-200", script, env, repo_root=Path(tmp))
            self.assertNotEqual(k1, k2)

    def test_different_ref_env_different_key(self) -> None:
        from evals._common import _ref_cache_key

        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "ref.sh"
            script.write_text("#!/bin/bash\necho hello\n", encoding="utf-8")
            k1 = _ref_cache_key(
                "long-train",
                script,
                {"WORLD_SIZE": "8"},
                repo_root=Path(tmp),
            )
            k2 = _ref_cache_key(
                "long-train",
                script,
                {"WORLD_SIZE": "4"},
                repo_root=Path(tmp),
            )
            self.assertNotEqual(k1, k2)

    def test_different_script_content_different_key(self) -> None:
        from evals._common import _ref_cache_key

        with tempfile.TemporaryDirectory() as tmp:
            s1 = Path(tmp) / "ref_v1.sh"
            s2 = Path(tmp) / "ref_v2.sh"
            s1.write_text("#!/bin/bash\necho v1\n", encoding="utf-8")
            s2.write_text("#!/bin/bash\necho v2\n", encoding="utf-8")
            env = {"WORLD_SIZE": "8"}
            k1 = _ref_cache_key("long-train", s1, env, repo_root=Path(tmp))
            k2 = _ref_cache_key("long-train", s2, env, repo_root=Path(tmp))
            self.assertNotEqual(k1, k2)

    def test_volatile_keys_excluded(self) -> None:
        from evals._common import _ref_cache_key

        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "ref.sh"
            script.write_text("#!/bin/bash\necho hello\n", encoding="utf-8")
            env_a = {"WORLD_SIZE": "8", "DUMP_DIR": "/run/1/dump"}
            env_b = {"WORLD_SIZE": "8", "DUMP_DIR": "/run/2/dump"}
            k1 = _ref_cache_key("long-train", script, env_a, repo_root=Path(tmp))
            k2 = _ref_cache_key("long-train", script, env_b, repo_root=Path(tmp))
            self.assertEqual(k1, k2)


class TestRefCacheKeyFingerprint(unittest.TestCase):
    """Cache key must also reflect the binary stack and source tree.

    Direct script-content hashing alone misses two real bug classes
    that hit loop e180edc7 repeatedly:

    1. The ref script ``source``s sibling shells under ``ref/reference/``;
       editing a sourced shell changes runtime behaviour without
       touching the script-content hash.
    2. CUDA / torch upgrades shift kernel selection and bit-level
       results; reusing a cache captured against an older stack
       produces phantom regressions in downstream bit-exact gates.

    Both are captured by ``_environment_fingerprint(repo_root)``,
    which the cache key must include.
    """

    def test_fingerprint_changes_invalidate_key(self) -> None:
        from evals import _common
        from evals._common import _ref_cache_key

        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "ref.sh"
            script.write_text("#!/bin/bash\necho hi\n", encoding="utf-8")
            env = {"WORLD_SIZE": "8"}

            with mock.patch.object(
                _common,
                "_environment_fingerprint",
                return_value="torch=2.5;git=aaa",
            ):
                k1 = _ref_cache_key("long-train", script, env, repo_root=Path(tmp))
            with mock.patch.object(
                _common,
                "_environment_fingerprint",
                return_value="torch=2.6;git=aaa",
            ):
                k2 = _ref_cache_key("long-train", script, env, repo_root=Path(tmp))
            with mock.patch.object(
                _common,
                "_environment_fingerprint",
                return_value="torch=2.5;git=bbb",
            ):
                k3 = _ref_cache_key("long-train", script, env, repo_root=Path(tmp))

            self.assertNotEqual(k1, k2, "torch version change must invalidate")
            self.assertNotEqual(k1, k3, "git sha change must invalidate")
            self.assertNotEqual(k2, k3)

    def test_fingerprint_includes_torch_version(self) -> None:
        from evals._common import _environment_fingerprint

        with tempfile.TemporaryDirectory() as tmp:
            fp = _environment_fingerprint(Path(tmp))
            # Either real torch version embedded, or the explicit
            # sentinel — never silently empty.
            self.assertTrue("torch=" in fp)
            self.assertNotIn("torch=;", fp)

    def test_fingerprint_includes_git_sha_when_available(self) -> None:
        from evals._common import _environment_fingerprint

        with tempfile.TemporaryDirectory() as tmp:
            fp = _environment_fingerprint(Path(tmp))
            # Outside a repo we expect the explicit ``no-git`` sentinel;
            # never silently empty — empty would let a non-repo dir
            # collide with any other non-repo dir at runtime.
            self.assertIn("git=", fp)
            self.assertNotIn("git=;", fp)
            self.assertFalse(fp.endswith("git="))

    def test_fingerprint_cached_per_repo(self) -> None:
        # The fingerprint involves a subprocess and may be hit on every
        # ref-script invocation; memoise per resolved repo path so the
        # gate dispatcher does not fork ``git`` once per suite.
        from evals import _common
        from evals._common import _environment_fingerprint

        with tempfile.TemporaryDirectory() as tmp:
            _common._FINGERPRINT_CACHE.clear()
            with mock.patch("subprocess.check_output") as m:
                m.return_value = "deadbeef\n"
                _environment_fingerprint(Path(tmp))
                _environment_fingerprint(Path(tmp))
                _environment_fingerprint(Path(tmp))
                self.assertLessEqual(m.call_count, 1)


class TestRefCacheSaveLoad(unittest.TestCase):
    """Round-trip of cache save and load."""

    def test_save_and_load_roundtrip(self) -> None:
        from evals._common import _save_ref_cache, _try_load_ref_cache

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache" / "abc123"
            dump_dir = Path(tmp) / "original_dump"
            ref_run = _make_ref_run(dump_dir, metadata={"num_steps": 200, "seed": 42})
            loss_by_step = {101: 5.0, 102: 4.8, 103: 4.6}
            grad_norm_by_step = {101: 1.5, 102: 1.4, 103: 1.3}

            _save_ref_cache(cache_dir, loss_by_step, grad_norm_by_step, ref_run)
            loaded = _try_load_ref_cache(cache_dir)

            self.assertIsNotNone(loaded)
            loaded_loss, loaded_grad, loaded_metadata, loaded_elapsed = loaded  # type: ignore[misc]
            self.assertEqual(loaded_loss, loss_by_step)
            self.assertEqual(loaded_grad, grad_norm_by_step)
            self.assertEqual(loaded_metadata["num_steps"], 200)
            self.assertEqual(loaded_metadata["seed"], 42)
            self.assertAlmostEqual(loaded_elapsed, 42.0)

    def test_legacy_cache_without_grad_loads_empty_grad(self) -> None:
        # A cache manifest written before grad_norm was persisted must
        # still load (empty grad baseline) — the bitwise gate then fails
        # fast on the empty baseline rather than the loader crashing.
        from evals._common import _try_load_ref_cache

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "legacy"
            cache_dir.mkdir()
            (cache_dir / "cache_manifest.json").write_text(
                '{"loss_by_step": {"101": 5.0}, "metadata": {"num_steps": 200}, "elapsed_s": 42.0}',
                encoding="utf-8",
            )
            loaded = _try_load_ref_cache(cache_dir)
            self.assertIsNotNone(loaded)
            loaded_loss, loaded_grad, _meta, _elapsed = loaded  # type: ignore[misc]
            self.assertEqual(loaded_loss, {101: 5.0})
            self.assertEqual(loaded_grad, {})

    def test_load_returns_none_on_missing_dir(self) -> None:
        from evals._common import _try_load_ref_cache

        with tempfile.TemporaryDirectory() as tmp:
            result = _try_load_ref_cache(Path(tmp) / "nonexistent")
            self.assertIsNone(result)

    def test_load_returns_none_on_corrupt_manifest(self) -> None:
        from evals._common import _try_load_ref_cache

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "corrupt"
            cache_dir.mkdir()
            (cache_dir / "cache_manifest.json").write_text("not json{{{", encoding="utf-8")
            result = _try_load_ref_cache(cache_dir)
            self.assertIsNone(result)


class TestRunViaRefScriptCaching(unittest.TestCase):
    """Integration: run_via_ref_script uses cache on second call."""

    def _setup_fake_repo(self, tmp: str) -> tuple[Path, Path, dict]:
        repo = Path(tmp)
        ref_dir = repo / "ref" / "reference"
        ref_dir.mkdir(parents=True)
        ref_script = ref_dir / "train_ref.sh"
        ref_script.write_text("#!/bin/bash\necho ref\n", encoding="utf-8")

        (repo / "config").mkdir(parents=True, exist_ok=True)
        artifact_dir = repo / ".artifacts" / "test_run"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        cfg = {
            "script": "eval.py",
            "launcher": "launch.py",
            "timeout_s": 600,
            "ref_env": {
                "WORLD_SIZE": "8",
                "NUM_STEPS_OVERRIDE": "200",
            },
        }
        return repo, artifact_dir, cfg

    def _mock_workload_config(self, repo: Path) -> dict:
        # run_via_ref_script resolves the ref timeout through
        # suite_ref_timeout_s, which requires the suite to exist in
        # [evals]; resolve_assets requires the [ref] tokenizer assets
        # (pre-populated dir → _ensure_tokenizer no-op, no HF download).
        tok_dir = repo / "tokenizer"
        tok_dir.mkdir(exist_ok=True)
        (tok_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
        return {
            "ref": {
                "ref_script": "train_ref.sh",
                "data_path": "/data/train",
                "forge_tokenizer_dir": str(tok_dir),
                "tokenizer": "dummy/tokenizer",
            },
            "env": {},
            "runtime": {"distributed": {"master_addr": "localhost", "master_port": "29500"}},
            "evals": {"long-train": {"timeout_s": 600}},
        }

    @mock.patch("evals._common._load_workload_config_for_ref")
    @mock.patch("evals._common._ref_script_runner")
    def test_cache_miss_then_hit(
        self,
        mock_runner_mod: mock.MagicMock,
        mock_load_config: mock.MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo, artifact_dir, cfg = self._setup_fake_repo(tmp)
            mock_load_config.return_value = self._mock_workload_config(repo)

            artifact_dir / "ref_dump__long-train"

            def fake_run_ref_script(**kwargs):
                dd = kwargs["dump_dir"]
                dd.mkdir(parents=True, exist_ok=True)
                loss_file = dd / "ref_loss.txt"
                loss_file.write_text(
                    "step=101 global_loss=5.0\nstep=102 global_loss=4.8\n",
                    encoding="utf-8",
                )
                return _make_ref_run(
                    dd,
                    loss_file=loss_file,
                    metadata={"gate_window_start": 101, "gate_window_end": 201, "num_steps": 200},
                )

            mock_runner_mod.run_ref_script.side_effect = fake_run_ref_script
            mock_runner_mod.parse_loss_dump.return_value = {101: 5.0, 102: 4.8}
            mock_runner_mod.parse_loss_dump_with_grad.return_value = {
                101: {"global_loss": 5.0, "grad_norm": 1.5},
                102: {"global_loss": 4.8, "grad_norm": 1.4},
            }
            mock_runner_mod.parse_stdout_loss.return_value = {}
            mock_runner_mod.RefRun = RefRun

            from evals._common import run_via_ref_script

            # First call: cache MISS — ref script runs
            result1 = run_via_ref_script(
                repo_root=repo,
                suite_key="long-train",
                cfg=cfg,
                artifact_dir=artifact_dir,
            )
            self.assertEqual(mock_runner_mod.run_ref_script.call_count, 1)
            self.assertEqual(result1["loss_by_step"], {101: 5.0, 102: 4.8})
            self.assertEqual(result1["grad_norm_by_step"], {101: 1.5, 102: 1.4})
            self.assertTrue(result1["ref_run"].succeeded)

            # Second call: cache HIT — ref script does NOT run again
            result2 = run_via_ref_script(
                repo_root=repo,
                suite_key="long-train",
                cfg=cfg,
                artifact_dir=artifact_dir,
            )
            self.assertEqual(mock_runner_mod.run_ref_script.call_count, 1)
            self.assertEqual(result2["loss_by_step"], {101: 5.0, 102: 4.8})
            # Cache HIT must replay the grad baseline, not just the loss.
            self.assertEqual(result2["grad_norm_by_step"], {101: 1.5, 102: 1.4})
            self.assertTrue(result2["ref_run"].succeeded)
            self.assertEqual(result2["ref_run"].metadata["num_steps"], 200)

    @mock.patch("evals._common._load_workload_config_for_ref")
    @mock.patch("evals._common._ref_script_runner")
    def test_failed_ref_run_not_cached(
        self,
        mock_runner_mod: mock.MagicMock,
        mock_load_config: mock.MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo, artifact_dir, cfg = self._setup_fake_repo(tmp)
            mock_load_config.return_value = self._mock_workload_config(repo)

            def fake_run_failing(**kwargs):
                dd = kwargs["dump_dir"]
                dd.mkdir(parents=True, exist_ok=True)
                return RefRun(
                    returncode=1,
                    stdout_path=dd / "ref_stdout.log",
                    dump_dir=dd,
                    loss_file=None,
                    elapsed_s=10.0,
                    timed_out=False,
                    metadata={},
                )

            mock_runner_mod.run_ref_script.side_effect = fake_run_failing
            mock_runner_mod.parse_loss_dump.return_value = {}
            mock_runner_mod.parse_stdout_loss.return_value = {}
            mock_runner_mod.RefRun = RefRun

            from evals._common import run_via_ref_script

            run_via_ref_script(
                repo_root=repo,
                suite_key="long-train",
                cfg=cfg,
                artifact_dir=artifact_dir,
            )
            # Second call should still invoke ref script (not cached)
            run_via_ref_script(
                repo_root=repo,
                suite_key="long-train",
                cfg=cfg,
                artifact_dir=artifact_dir,
            )
            self.assertEqual(mock_runner_mod.run_ref_script.call_count, 2)

    @mock.patch("evals._common._load_workload_config_for_ref")
    @mock.patch("evals._common._ref_script_runner")
    def test_cache_invalidated_on_ref_env_change(
        self,
        mock_runner_mod: mock.MagicMock,
        mock_load_config: mock.MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo, artifact_dir, cfg = self._setup_fake_repo(tmp)
            mock_load_config.return_value = self._mock_workload_config(repo)

            call_count = [0]

            def fake_run(**kwargs):
                call_count[0] += 1
                dd = kwargs["dump_dir"]
                dd.mkdir(parents=True, exist_ok=True)
                loss_file = dd / "ref_loss.txt"
                loss_file.write_text("step=101 global_loss=5.0\n", encoding="utf-8")
                return _make_ref_run(dd, loss_file=loss_file)

            mock_runner_mod.run_ref_script.side_effect = fake_run
            mock_runner_mod.parse_loss_dump.return_value = {101: 5.0}
            mock_runner_mod.parse_loss_dump_with_grad.return_value = {
                101: {"global_loss": 5.0, "grad_norm": 1.5}
            }
            mock_runner_mod.parse_stdout_loss.return_value = {}
            mock_runner_mod.RefRun = RefRun

            from evals._common import run_via_ref_script

            # First call with WORLD_SIZE=8. The cfg-level ref_env overlay
            # died with the gate-product collapse — the caller extra_env
            # channel is what feeds merged_extra_env (and the cache key).
            run_via_ref_script(
                repo_root=repo,
                suite_key="long-train",
                cfg=cfg,
                artifact_dir=artifact_dir,
                extra_env={"WORLD_SIZE": "8"},
            )
            self.assertEqual(call_count[0], 1)

            # Change extra_env → cache key changes → ref script re-runs
            run_via_ref_script(
                repo_root=repo,
                suite_key="long-train",
                cfg=cfg,
                artifact_dir=artifact_dir,
                extra_env={"WORLD_SIZE": "4"},
            )
            self.assertEqual(call_count[0], 2)


class TestRefCacheDisable(unittest.TestCase):
    """FORGE_REF_CACHE=0 disables caching."""

    @mock.patch("evals._common._load_workload_config_for_ref")
    @mock.patch("evals._common._ref_script_runner")
    def test_env_disables_cache(
        self,
        mock_runner_mod: mock.MagicMock,
        mock_load_config: mock.MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            ref_dir = repo / "ref" / "reference"
            ref_dir.mkdir(parents=True)
            (ref_dir / "train_ref.sh").write_text("#!/bin/bash\necho ref\n", encoding="utf-8")
            artifact_dir = repo / ".artifacts" / "test"
            artifact_dir.mkdir(parents=True, exist_ok=True)

            cfg = {"timeout_s": 60, "ref_env": {"WORLD_SIZE": "8"}}
            tok_dir = repo / "tokenizer"
            tok_dir.mkdir()
            (tok_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
            mock_load_config.return_value = {
                "ref": {
                    "ref_script": "train_ref.sh",
                    "forge_tokenizer_dir": str(tok_dir),
                    "tokenizer": "dummy/tokenizer",
                },
                "env": {},
                "evals": {"long-train": {"timeout_s": 60}},
            }

            def fake_run(**kwargs):
                dd = kwargs["dump_dir"]
                dd.mkdir(parents=True, exist_ok=True)
                return _make_ref_run(dd)

            mock_runner_mod.run_ref_script.side_effect = fake_run
            mock_runner_mod.parse_loss_dump.return_value = {}
            mock_runner_mod.parse_stdout_loss.return_value = {101: 5.0}
            mock_runner_mod.RefRun = RefRun

            from evals._common import run_via_ref_script

            with mock.patch.dict("os.environ", {"FORGE_REF_CACHE": "0"}):
                run_via_ref_script(
                    repo_root=repo,
                    suite_key="long-train",
                    cfg=cfg,
                    artifact_dir=artifact_dir,
                )
                run_via_ref_script(
                    repo_root=repo,
                    suite_key="long-train",
                    cfg=cfg,
                    artifact_dir=artifact_dir,
                )
            self.assertEqual(mock_runner_mod.run_ref_script.call_count, 2)


if __name__ == "__main__":
    unittest.main()

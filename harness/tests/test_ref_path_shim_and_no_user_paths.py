"""Single-machine path elimination contracts (Issue: zhuzhui hardcode purge).

Two principles are pinned here, both flowing from the dense_training.toml SSOT
review on 2026-05-17:

1. **Real deployment paths live in dense_training.toml only.** Anything that a
   user genuinely has to fill in for their machine — the Megatron source
   root, the Megatron-binary data prefix, the canonical-checkpoint root,
   the SentencePiece tokenizer.model — is owned by
   ``[defaults]`` of ``config/eval.toml`` and surfaced through the
   standard env / CLI override chain.
2. **No file in the tree binds to a single machine.** Every text
   candidate under the harness write surface (and the train_engine job
   specs) is forbidden from referencing ``/user/zhuzhui``,
   ``/user/qiqi`` or ``/user/chenyaojian`` etc. — historical author dirs
   that broke fresh-machine bring-up.

The first concern is exercised by the env-shim contract tests; the
second is enforced by a static repo-wide scan.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
GIT_TOPLEVEL = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ──────────────────────────────────────────────────────────────────────
# 1. Env-shim contract: harness pushes deployment + gate-private paths
#    into the L0 ref-script subprocess so the script has zero hardcoded
#    machine-specific defaults of its own.
# ──────────────────────────────────────────────────────────────────────


class _FakeRefRun:
    def __init__(self, *, succeeded: bool = True, metadata: dict | None = None) -> None:
        self.succeeded = succeeded
        self.metadata = metadata or {}
        self.loss_file = None
        self.stdout_path = Path("/dev/null")
        self.dump_dir = Path("/dev/null")


def _seed_dummy_ref_script(repo_root: Path) -> Path:
    ref_dir = repo_root / "ref" / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    script = ref_dir / "train_minicpm4_0.5b_gsm8k.sh"
    script.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    return script


class TestRefScriptPathShim(unittest.TestCase):
    """``_ref_script_path_env`` is the SSOT for "what paths the harness
    pushes into the L0 ref script subprocess". Five keys are required:

    * ``MEGATRON_ROOT`` / ``DATA_PATH`` / ``TOKENIZER_MODEL`` —
      deployment-config from ``dense_training.toml [defaults]``.
    * ``SAVE_PATH`` / ``TENSORBOARD_DIR`` — gate-private artifact paths
      auto-derived from ``dump_dir`` so two concurrent gates never
      collide and so the scripts never fall back to a personal default.
    """

    def test_emits_deployment_paths_from_workload_defaults(self) -> None:
        from evals._common import _ref_script_path_env

        wc = {
            "ref": {
                "megatron_root": "/opt/Megatron-LM",
                "data_path": "/srv/data/gsm8k_megatron/gsm8k_train_text_document",
                "tokenizer_model": "/srv/models/MiniCPM4-0.5B/tokenizer.model",
            }
        }
        env = _ref_script_path_env(REPO_ROOT, {}, workload_config=wc)
        self.assertEqual(env["MEGATRON_ROOT"], "/opt/Megatron-LM")
        self.assertEqual(
            env["DATA_PATH"],
            "/srv/data/gsm8k_megatron/gsm8k_train_text_document",
        )
        self.assertEqual(
            env["TOKENIZER_MODEL"],
            "/srv/models/MiniCPM4-0.5B/tokenizer.model",
        )

    def test_emits_gate_private_save_and_tensorboard_when_dump_dir_provided(
        self,
    ) -> None:
        from evals._common import _ref_script_path_env

        with tempfile.TemporaryDirectory() as tmp:
            dump_dir = Path(tmp) / "ref_dump__multistep-1gpu"
            env = _ref_script_path_env(
                REPO_ROOT,
                {},
                workload_config={"ref": {}},
                dump_dir=dump_dir,
            )
            # SAVE_PATH and TENSORBOARD_DIR are gate-private subpaths of
            # dump_dir so two concurrent gates never write to the same
            # tensorboard or save dir, and so neither defaults to a
            # user-specific filesystem location.
            self.assertEqual(env["SAVE_PATH"], str(dump_dir / "save"))
            self.assertEqual(env["TENSORBOARD_DIR"], str(dump_dir / "tensorboard"))

    def test_omits_save_and_tensorboard_when_no_dump_dir(self) -> None:
        from evals._common import _ref_script_path_env

        env = _ref_script_path_env(
            REPO_ROOT,
            {},
            workload_config={"ref": {}},
        )
        self.assertNotIn("SAVE_PATH", env)
        self.assertNotIn("TENSORBOARD_DIR", env)

    def test_skips_keys_with_empty_values(self) -> None:
        from evals._common import _ref_script_path_env

        wc = {
            "ref": {
                "megatron_root": "",
                "data_path": "",
                "tokenizer_model": "",
            }
        }
        env = _ref_script_path_env(REPO_ROOT, {}, workload_config=wc)
        # Empty string in toml means "not set" (user has not configured
        # this machine); env shim must NOT inject empty values, otherwise
        # the ref-script's ``${VAR:?...}`` fail-fast guards would be
        # masked by a present-but-empty environment variable.
        self.assertNotIn("MEGATRON_ROOT", env)
        self.assertNotIn("DATA_PATH", env)
        self.assertNotIn("TOKENIZER_MODEL", env)


class TestRunViaRefScriptPropagatesAllPathShims(unittest.TestCase):
    """End-to-end: ``run_via_ref_script`` flows TOKENIZER_MODEL +
    gate-private SAVE_PATH/TENSORBOARD_DIR all the way to the
    ref-script subprocess env."""

    def test_tokenizer_and_gate_private_paths_reach_subprocess(self) -> None:
        from evals import _common

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            _seed_dummy_ref_script(repo_root)
            # Pre-populated tokenizer dir so resolve_assets is a cache hit
            # (no HF download in unit tests).
            tok_dir = repo_root / "assets" / "tokenizer"
            tok_dir.mkdir(parents=True, exist_ok=True)
            (tok_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
            (repo_root / "config").mkdir(parents=True, exist_ok=True)
            (repo_root / "config" / "eval.toml").write_text(
                "[workload]\n"
                'id = "x"\n'
                'display_name = "x"\n'
                "requires_cuda = false\n"
                "[env]\n"
                "[runtime.distributed]\n"
                'master_addr = "localhost"\n'
                'master_port = "29500"\n'
                "[ref]\n"
                'megatron_root = "/opt/mg"\n'
                'data_path = "/srv/data/p"\n'
                'tokenizer_model = "/srv/tok.model"\n'
                'tokenizer = "org/dummy-tokenizer"\n'
                f'forge_tokenizer_dir = "{tok_dir}"\n'
                'ref_script = "train_minicpm4_0.5b_gsm8k.sh"\n'
                'ref_capture_script = ""\n'
                'ref_capture_basename = "ref_capture.pt"\n'
                'candidate_capture_basename = "candidate_capture.pt"\n'
                "[evals.multistep-1gpu]\n"
                'stage = "stage1"\n'
                'runner_kind = "stage1-bitwise-trajectory"\n'
                'env_inputs = ["DATA_PATH"]\n',
                encoding="utf-8",
            )

            artifact_dir = repo_root / ".artifacts"
            captured: dict[str, dict[str, str]] = {}

            def fake_run_ref_script(**kwargs: object) -> _FakeRefRun:
                captured["extra_env"] = kwargs["extra_env"]  # type: ignore[assignment]
                return _FakeRefRun(metadata={"num_steps": 1})

            with (
                mock.patch(
                    "tools.ref_script_runner.run_ref_script",
                    side_effect=fake_run_ref_script,
                ),
                mock.patch("tools.ref_script_runner.parse_loss_dump", return_value={}),
                mock.patch("tools.ref_script_runner.parse_stdout_loss", return_value={}),
            ):
                _common.run_via_ref_script(
                    repo_root=repo_root,
                    suite_key="multistep-1gpu",
                    cfg={},
                    artifact_dir=artifact_dir,
                    capture_loss_trace=False,
                )

        env = captured["extra_env"]
        self.assertEqual(env["MEGATRON_ROOT"], "/opt/mg")
        self.assertEqual(env["DATA_PATH"], "/srv/data/p")
        # Tokenizer now flows via the resolve_assets channel
        # (FORGE_TOKENIZER_DIR), not a TOKENIZER_MODEL file path.
        self.assertEqual(env["FORGE_TOKENIZER_DIR"], str(tok_dir))
        # Gate-private dump dir uses suite_key — never a user dir.
        expected_dump = artifact_dir / "ref_dump__multistep-1gpu"
        self.assertEqual(env["SAVE_PATH"], str(expected_dump / "save"))
        self.assertEqual(env["TENSORBOARD_DIR"], str(expected_dump / "tensorboard"))


# ──────────────────────────────────────────────────────────────────────
# 2. Tokenizer model env override (FORGE_TOKENIZER_MODEL).
# ──────────────────────────────────────────────────────────────────────


class TestTokenizerModelEnvOverride(unittest.TestCase):
    def test_env_override_lifts_into_workload_defaults(self) -> None:
        import os

        from harness import config_runtime

        prior = os.environ.get("FORGE_TOKENIZER_MODEL")
        try:
            os.environ["FORGE_TOKENIZER_MODEL"] = "/srv/custom-tok.model"
            data: dict[str, object] = {"ref": {"tokenizer_model": ""}}
            config_runtime._apply_ref_env_overrides(data)
            self.assertEqual(
                data["ref"]["tokenizer_model"],  # type: ignore[index]
                "/srv/custom-tok.model",
            )
        finally:
            if prior is None:
                os.environ.pop("FORGE_TOKENIZER_MODEL", None)
            else:
                os.environ["FORGE_TOKENIZER_MODEL"] = prior


# ──────────────────────────────────────────────────────────────────────
# 3. Repo-wide scan: forbid hardcoded individual-user absolute paths.
# ──────────────────────────────────────────────────────────────────────


# Banned absolute prefixes. Each entry is the literal substring we refuse
# to find anywhere in the tracked tree (case-sensitive — these are
# Linux-style absolute paths so case matters).
BANNED_USER_PATH_PREFIXES = (
    "/user/zhuzhui",
    "/user/qiqi",
    "/user/chenyaojian",
)

# Files that we deliberately allow to mention these strings (this very
# test file is the canonical example — it spells the banned prefixes out
# as data, not as machine-bound paths). Add new entries only when the
# file genuinely needs to talk about the historical paths (e.g. a
# changelog explaining the migration).
ALLOWLIST_RELATIVE = frozenset(
    {
        # The scan test itself.
        "harness/harness/tests/test_ref_path_shim_and_no_user_paths.py",
        # Documentation uses example paths illustratively.
        "harness/prompt/develop_prompt/remote-execution.md",
    }
)

# Suffixes scanned. Mirrors framework_guard.TEXT_SUFFIXES but kept local
# so this test does not couple to that module's set.
SCAN_SUFFIXES = frozenset(
    {
        ".cfg",
        ".ini",
        ".json",
        ".jsonl",
        ".md",
        ".mdc",
        ".py",
        ".rst",
        ".sh",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
)
SCAN_BASENAMES = frozenset({".dockerignore", ".gitignore", "Dockerfile", "Makefile"})
PRUNED_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".artifacts",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".tox",
        "node_modules",
        "third_party",
    }
)


def _iter_text_files(root: Path):
    output = subprocess.check_output(
        ["git", "-C", str(root), "ls-files"],
        text=True,
    )
    for rel_text in output.splitlines():
        path = root / rel_text
        if any(part in PRUNED_DIRS for part in path.parts):
            continue
        if path.suffix.lower() in SCAN_SUFFIXES or path.name in SCAN_BASENAMES:
            yield path


class TestNoMachineBoundPaths(unittest.TestCase):
    def test_no_user_specific_absolute_paths(self) -> None:
        # Scan tracked files only. Gitignored local profiles such as
        # config/ref.toml are deployment-local copies and may contain
        # machine paths that must not affect unit tests.
        scan_root = GIT_TOPLEVEL
        violations: list[tuple[str, int, str]] = []
        pattern = re.compile("|".join(re.escape(p) for p in BANNED_USER_PATH_PREFIXES))
        for path in _iter_text_files(scan_root):
            try:
                rel = path.resolve().relative_to(scan_root.resolve())
            except ValueError:
                continue
            if str(rel).replace("\\", "/") in ALLOWLIST_RELATIVE:
                continue
            try:
                contents = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                # Tracked-but-deleted working copies (mid-refactor) and
                # binary-ish files are skipped, not violations.
                continue
            for line_no, line in enumerate(contents.splitlines(), start=1):
                if pattern.search(line):
                    violations.append((str(rel), line_no, line.strip()))

        if violations:
            rendered = "\n".join(
                f"  {rel}:{line_no}: {line}" for rel, line_no, line in violations[:25]
            )
            self.fail(
                "Hardcoded individual-user absolute path(s) found "
                "(machine-bound; replace with dense_training.toml SSOT or "
                "auto-derived path):\n" + rendered
            )


# ──────────────────────────────────────────────────────────────────────
# 4. Static check: the ref scripts ship a repo-relative auto-derive
#    fallback for SAVE_PATH / TENSORBOARD_DIR, so an ad-hoc
#    ``bash ref/reference/<script>.sh`` invocation never falls back to
#    a personal directory.
# ──────────────────────────────────────────────────────────────────────


class TestRefScriptAutoDerivedSavePaths(unittest.TestCase):
    def test_gsm8k_script_has_no_user_paths(self) -> None:
        script = REPO_ROOT / "ref" / "reference" / "train_minicpm4_0.5b_gsm8k.sh"
        text = script.read_text(encoding="utf-8")
        for prefix in BANNED_USER_PATH_PREFIXES:
            self.assertNotIn(
                prefix,
                text,
                f"L0 ref script must not embed {prefix!r}",
            )

    def test_gsm8k_script_save_path_falls_back_to_repo_relative(self) -> None:
        script = REPO_ROOT / "ref" / "reference" / "train_minicpm4_0.5b_gsm8k.sh"
        text = script.read_text(encoding="utf-8")
        # The auto-derived fallback uses the .artifacts/ref_local subtree
        # so a fresh-machine bash invocation lands in-repo (gitignored).
        self.assertIn(".artifacts/ref_local", text)

    def test_fineweb_script_has_no_user_paths(self) -> None:
        script = REPO_ROOT / "ref" / "reference" / "train_minicpm4_0.5b_fineweb_modelbestsdk.sh"
        text = script.read_text(encoding="utf-8")
        for prefix in BANNED_USER_PATH_PREFIXES:
            self.assertNotIn(prefix, text)

    def test_prepare_script_has_no_user_paths(self) -> None:
        script = REPO_ROOT / "ref" / "reference" / "prepare_gsm8k_data.sh"
        text = script.read_text(encoding="utf-8")
        for prefix in BANNED_USER_PATH_PREFIXES:
            self.assertNotIn(prefix, text)


# ──────────────────────────────────────────────────────────────────────
# 5. dense_training.toml [defaults] contains only task-level keys — deployment
#    paths live in config/ref.toml (deployment-level separation).
# ──────────────────────────────────────────────────────────────────────


class TestWorkloadRefHasNoMachineBoundValues(unittest.TestCase):
    def test_ref_paths_are_not_in_task_level_config(self) -> None:
        from harness._compat import tomllib

        config_path = REPO_ROOT / "config" / "eval" / "dense_training" / "dense_training.toml"
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
        ref = data["ref"]
        run_level_keys = ("megatron_root", "checkpoint_root", "data_path", "tokenizer_model")
        for key in run_level_keys:
            self.assertNotIn(
                key,
                ref,
                f"[ref].{key} is a deployment-level path and must not appear "
                "in dense_training.toml — move to config/ref/{{profile}}.toml",
            )

    def test_ref_base_templates_exist(self) -> None:
        ref_dir = REPO_ROOT / "config" / "ref"
        self.assertTrue(ref_dir.is_dir(), "config/ref/ directory must exist")
        self.assertTrue(
            (ref_dir / "megatron_minicpm4_0.5b.toml").is_file(),
            "config/ref/megatron_minicpm4_0.5b.toml must exist (base template)",
        )
        self.assertTrue(
            (ref_dir / "torch_minicpm4_0.5b.toml").is_file(),
            "config/ref/torch_minicpm4_0.5b.toml must exist (base template)",
        )


if __name__ == "__main__":
    unittest.main()

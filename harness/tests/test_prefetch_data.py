"""Tests for tools/prefetch_data.py pure logic (no network).

Covers the part-series expansion, en/zh interleave, sentinel I/O, and the
production-milestone readiness wait. The HF-hub download in ``main`` is network-bound and
intentionally not exercised here.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools import prefetch_data as pf


class TestEnumeratePartsFromSeed(unittest.TestCase):
    def test_expands_full_series_preserving_width(self) -> None:
        parts = pf.enumerate_parts_from_seed(
            "en/ultrafineweb-en-part-0001-of-2048.parquet",
            "data/ultrafineweb_en/ultrafineweb-en-part-0001-of-2048.parquet",
        )
        self.assertEqual(len(parts), 2048)
        # First / last keep the 4-digit zero pad and the -of-2048 tail.
        self.assertEqual(
            parts[0],
            (
                "en/ultrafineweb-en-part-0001-of-2048.parquet",
                "data/ultrafineweb_en/ultrafineweb-en-part-0001-of-2048.parquet",
            ),
        )
        self.assertEqual(parts[-1][0], "en/ultrafineweb-en-part-2048-of-2048.parquet")
        self.assertEqual(
            parts[-1][1],
            "data/ultrafineweb_en/ultrafineweb-en-part-2048-of-2048.parquet",
        )

    def test_zh_three_digit_width(self) -> None:
        parts = pf.enumerate_parts_from_seed(
            "zh/ultrafineweb-zh-part-001-of-256.parquet",
            "data/ultrafineweb_zh/ultrafineweb-zh-part-001-of-256.parquet",
        )
        self.assertEqual(len(parts), 256)
        self.assertEqual(parts[0][0], "zh/ultrafineweb-zh-part-001-of-256.parquet")
        self.assertEqual(parts[9][0], "zh/ultrafineweb-zh-part-010-of-256.parquet")

    def test_rejects_seed_without_part_suffix(self) -> None:
        with self.assertRaises(ValueError):
            pf.enumerate_parts_from_seed("en/whatever.parquet", "data/whatever.parquet")


class TestInterleave(unittest.TestCase):
    def _en(self, n):
        return [(f"en/{i}", f"r/en/{i}") for i in range(n)]

    def _zh(self, n):
        return [(f"zh/{i}", f"r/zh/{i}") for i in range(n)]

    def test_default_fraction_is_mostly_english(self) -> None:
        order = pf.interleave(self._en(100), self._zh(100), 0.9)
        first20 = [k for k, _ in order[:20]]
        en = sum(1 for k in first20 if k.startswith("en/"))
        # ~90% English in the prefix, so the GB cutoff lands on a mix.
        self.assertGreaterEqual(en, 16)
        self.assertLessEqual(en, 20)

    def test_fraction_one_is_all_english_until_exhausted(self) -> None:
        order = pf.interleave(self._en(3), self._zh(5), 1.0)
        self.assertEqual([k for k, _ in order[:3]], ["en/0", "en/1", "en/2"])
        # zh only after en is exhausted.
        self.assertTrue(all(k.startswith("zh/") for k, _ in order[3:]))
        self.assertEqual(len(order), 8)

    def test_fraction_zero_is_all_zh_first(self) -> None:
        order = pf.interleave(self._en(2), self._zh(2), 0.0)
        self.assertEqual([k for k, _ in order[:2]], ["zh/0", "zh/1"])

    def test_all_parts_present_exactly_once(self) -> None:
        order = pf.interleave(self._en(7), self._zh(4), 0.6)
        self.assertEqual(len(order), 11)
        self.assertEqual(len({k for k, _ in order}), 11)

    def test_rejects_out_of_range_fraction(self) -> None:
        with self.assertRaises(ValueError):
            pf.interleave(self._en(1), self._zh(1), 1.5)


class TestSentinelIO(unittest.TestCase):
    def test_round_trip(self) -> None:
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            self.assertIsNone(pf.read_prefetch_status(d))
            pf.write_prefetch_status(d, {"status": "ok", "bytes": 123})
            got = pf.read_prefetch_status(d)
            self.assertEqual(got["status"], "ok")
            self.assertEqual(got["bytes"], 123)
            # Sentinel is the documented filename, atomically replaced.
            self.assertTrue((d / pf.PREFETCH_SENTINEL).is_file())
            self.assertFalse((d / (pf.PREFETCH_SENTINEL + ".tmp")).exists())

    def test_write_creates_missing_dir(self) -> None:
        with TemporaryDirectory() as tmp:
            d = Path(tmp) / "nested" / "forge"
            pf.write_prefetch_status(d, {"status": "ok"})
            self.assertEqual(json.loads((d / pf.PREFETCH_SENTINEL).read_text())["status"], "ok")


class TestDownloadUntil(unittest.TestCase):
    def _fetch_factory(self, size: int):
        def fetch(repo_path: str, dest: Path) -> None:
            dest.write_bytes(b"x" * size)

        return fetch

    def test_stops_once_target_reached(self) -> None:
        with TemporaryDirectory() as tmp:
            order = [(f"en/{i}.parquet", f"r/{i}") for i in range(100)]
            total, staged = pf.download_until(
                "repo", Path(tmp), order, target_bytes=25, fetch=self._fetch_factory(10)
            )
            # 10+10+10 = 30 ≥ 25 → 3 parts, then stop.
            self.assertEqual(staged, 3)
            self.assertEqual(total, 30)

    def test_existing_files_count_and_are_not_refetched(self) -> None:
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "en").mkdir()
            (d / "en" / "0.parquet").write_bytes(b"y" * 10)  # pre-existing
            calls = []

            def fetch(repo_path, dest):
                calls.append(repo_path)
                dest.write_bytes(b"x" * 10)

            order = [(f"en/{i}.parquet", f"r/{i}") for i in range(5)]
            total, staged = pf.download_until("repo", d, order, target_bytes=25, fetch=fetch)
            # part 0 already present (counts 10, not refetched); parts 1,2 fetched.
            self.assertNotIn("r/0", calls)
            self.assertEqual(staged, 2)
            self.assertEqual(total, 30)

    def test_progress_callback_fires_per_new_part(self) -> None:
        with TemporaryDirectory() as tmp:
            seen = []
            order = [(f"en/{i}.parquet", f"r/{i}") for i in range(3)]
            pf.download_until(
                "repo",
                Path(tmp),
                order,
                target_bytes=10**9,
                fetch=self._fetch_factory(5),
                on_progress=lambda s, t: seen.append((s, t)),
            )
            self.assertEqual(seen, [(1, 5), (2, 10), (3, 15)])


class TestCurlFetch(unittest.TestCase):
    def test_builds_resolve_url_from_endpoint_and_writes_file(self) -> None:
        import subprocess as _sp
        from unittest import mock

        captured = {}

        def fake_run(cmd, **kw):
            captured["cmd"] = cmd
            # curl's -o target is the last arg; create it so the is_file check passes.
            Path(cmd[-2]).write_bytes(b"data")
            return _sp.CompletedProcess(cmd, 0)

        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "en" / "p.parquet"
            dest.parent.mkdir(parents=True)
            with mock.patch("subprocess.run", side_effect=fake_run):
                fetch = pf._curl_fetch("openbmb/Ultra-FineWeb", endpoint="https://hf-mirror.com")
                fetch("data/ultrafineweb_en/x.parquet", dest)
            url = captured["cmd"][-1]
            self.assertEqual(
                url,
                "https://hf-mirror.com/datasets/openbmb/Ultra-FineWeb/resolve/main/"
                "data/ultrafineweb_en/x.parquet",
            )
            self.assertTrue(dest.is_file())

    def test_raises_on_curl_failure(self) -> None:
        import subprocess as _sp
        from unittest import mock

        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "p.parquet"
            with mock.patch("subprocess.run", return_value=_sp.CompletedProcess(["curl"], 6)):
                fetch = pf._curl_fetch("repo", endpoint="https://m")
                with self.assertRaises(RuntimeError):
                    fetch("a/b.parquet", dest)


class TestWaitForPrefetch(unittest.TestCase):
    def test_returns_when_ok(self) -> None:
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            pf.write_prefetch_status(d, {"status": "ok", "bytes": 9})
            status = pf.wait_for_prefetch(d, timeout_s=1, poll_s=0.01)
            self.assertEqual(status["bytes"], 9)

    def test_raises_on_error_status(self) -> None:
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            pf.write_prefetch_status(d, {"status": "error", "error": "boom"})
            with self.assertRaises(RuntimeError) as cm:
                pf.wait_for_prefetch(d, timeout_s=1, poll_s=0.01)
            self.assertIn("boom", str(cm.exception))

    def test_times_out_when_no_sentinel(self) -> None:
        # Injected clock: first read None, then deadline passes → RuntimeError.
        ticks = iter([0.0, 0.0, 100.0])
        with TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError) as cm:
                pf.wait_for_prefetch(
                    Path(tmp),
                    timeout_s=10,
                    poll_s=0.0,
                    sleep=lambda _s: None,
                    monotonic=lambda: next(ticks),
                )
            self.assertIn("did not finish", str(cm.exception))

    def test_polls_until_sentinel_appears(self) -> None:
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            calls = {"n": 0}

            def fake_sleep(_s):
                calls["n"] += 1
                if calls["n"] == 2:
                    pf.write_prefetch_status(d, {"status": "ok", "bytes": 1})

            status = pf.wait_for_prefetch(
                d,
                timeout_s=10,
                poll_s=0.0,
                sleep=fake_sleep,
                monotonic=lambda: 0.0,
            )
            self.assertEqual(status["status"], "ok")
            self.assertGreaterEqual(calls["n"], 2)


class TestForgeDataDirResolution(unittest.TestCase):
    """A relative ``[data].forge_data_dir`` must resolve against the
    ``--repo-root`` (the workspace), NOT the process cwd — so the corpus
    lands under the workspace tree wherever the prefetch runs, matching
    where the dispatcher waits for the sentinel and where the dataloader
    reads via ``FORGE_DATA_DIR``."""

    _DATA = {
        "dataset": "openbmb/Ultra-FineWeb",
        "prefetch_target_gb": 1,
        "download_files": {
            "en/ultrafineweb-en-part-0001-of-2048.parquet": "en/x.parquet",
            "zh/ultrafineweb-zh-part-001-of-256.parquet": "zh/y.parquet",
        },
    }

    def _run_capturing_target(self, repo_root: str, forge_data_dir: str) -> Path:
        from unittest import mock

        captured: dict[str, Path] = {}

        def fake_download(repo, fdd, order, target_bytes, fetch=None, on_progress=None):
            captured["fdd"] = fdd
            return (target_bytes, 0)

        data = {**self._DATA, "forge_data_dir": forge_data_dir}
        with (
            mock.patch.object(pf, "_data_axis", return_value=data),
            mock.patch.object(pf, "download_until", side_effect=fake_download),
            mock.patch.object(pf, "write_prefetch_status"),
        ):
            rc = pf.main(["--repo-root", repo_root])
        self.assertEqual(rc, 0)
        return captured["fdd"]

    def test_relative_resolves_against_repo_root_not_cwd(self) -> None:
        fdd = self._run_capturing_target("/repo/ws", ".artifacts/forge-data/uf")
        self.assertEqual(fdd, (Path("/repo/ws") / ".artifacts/forge-data/uf").resolve())

    def test_absolute_is_left_untouched(self) -> None:
        fdd = self._run_capturing_target("/repo/ws", "/opt/forge-data/uf")
        self.assertEqual(fdd, Path("/opt/forge-data/uf"))


if __name__ == "__main__":
    unittest.main()

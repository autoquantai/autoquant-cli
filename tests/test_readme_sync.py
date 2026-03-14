from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autoquant_cli.readme_sync import ensure_setup_repo_synced, get_readme_diff, run_update, sync_setup_repo


class ReadmeSyncTest(unittest.TestCase):
    def test_ensure_setup_repo_synced_clones_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            def fake_run(
                command: list[str],
                cwd: Path | None = None,
                capture_output: bool = True,
                text: bool = True,
                check: bool = False,
            ) -> subprocess.CompletedProcess[str]:
                if command[1] == "clone":
                    repo_dir = Path(command[-1])
                    repo_dir.mkdir(parents=True, exist_ok=True)
                    (repo_dir / ".git").mkdir(parents=True, exist_ok=True)
                    return subprocess.CompletedProcess(command, 0, "", "")
                if command[1] == "fetch":
                    return subprocess.CompletedProcess(command, 0, "", "")
                if command[1] == "rev-parse":
                    if command[2] == "HEAD":
                        return subprocess.CompletedProcess(command, 0, "abc123\n", "")
                    return subprocess.CompletedProcess(command, 0, "def456\n", "")
                return subprocess.CompletedProcess(command, 0, "", "")

            with patch.dict(os.environ, {"HOME": tmp_dir}, clear=False):
                with patch("autoquant_cli.readme_sync.subprocess.run", side_effect=fake_run):
                    result = ensure_setup_repo_synced()
                    self.assertEqual(result["baseline_commit"], "abc123")
                    self.assertEqual(result["latest_commit"], "def456")
                    self.assertTrue((Path(result["repo_dir"]) / ".git").exists())

    def test_ensure_setup_repo_synced_raises_when_fetch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_dir = Path(tmp_dir) / ".nanobot" / "workspace" / "autoquant" / "autoquant-setup"
            repo_dir.mkdir(parents=True, exist_ok=True)
            (repo_dir / ".git").mkdir(parents=True, exist_ok=True)

            def fake_run(
                command: list[str],
                cwd: Path | None = None,
                capture_output: bool = True,
                text: bool = True,
                check: bool = False,
            ) -> subprocess.CompletedProcess[str]:
                if command[1] == "fetch":
                    return subprocess.CompletedProcess(command, 1, "", "fetch failed")
                return subprocess.CompletedProcess(command, 0, "", "")

            with patch.dict(os.environ, {"HOME": tmp_dir}, clear=False):
                with patch("autoquant_cli.readme_sync.subprocess.run", side_effect=fake_run):
                    with self.assertRaises(RuntimeError) as context:
                        ensure_setup_repo_synced()
        self.assertIn("git fetch failed", str(context.exception))

    def test_sync_setup_repo_fast_forwards_when_behind(self) -> None:
        sync_state = {
            "repo_dir": "/tmp/autoquant-setup",
            "repo_url": "https://github.com/autoquantai/autoquant-setup.git",
            "branch": "main",
            "baseline_commit": "old_head",
            "latest_commit": "new_head",
        }
        with patch("autoquant_cli.readme_sync.ensure_setup_repo_synced", return_value=sync_state):
            with patch("autoquant_cli.readme_sync.fast_forward_setup_repo", return_value="new_head"):
                result = sync_setup_repo()
        self.assertTrue(result["updated"])
        self.assertEqual(result["head_after_sync"], "new_head")

    def test_sync_setup_repo_noop_when_up_to_date(self) -> None:
        sync_state = {
            "repo_dir": "/tmp/autoquant-setup",
            "repo_url": "https://github.com/autoquantai/autoquant-setup.git",
            "branch": "main",
            "baseline_commit": "same_head",
            "latest_commit": "same_head",
        }
        with patch("autoquant_cli.readme_sync.ensure_setup_repo_synced", return_value=sync_state):
            with patch("autoquant_cli.readme_sync.fast_forward_setup_repo") as fast_forward_mock:
                result = sync_setup_repo()
        self.assertFalse(result["updated"])
        self.assertEqual(result["head_after_sync"], "same_head")
        fast_forward_mock.assert_not_called()

    def test_run_update_syncs_repo_and_installs_package(self) -> None:
        sync_result = {
            "repo_dir": "/tmp/autoquant-setup",
            "repo_url": "https://github.com/autoquantai/autoquant-setup.git",
            "branch": "main",
            "baseline_commit": "old_head",
            "latest_commit": "new_head",
            "head_after_sync": "new_head",
            "updated": True,
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            pip_path = Path(tmp_dir) / ".nanobot" / "workspace" / "autoquant" / "venv" / "autoquant" / "bin" / "pip"
            pip_path.parent.mkdir(parents=True, exist_ok=True)
            pip_path.write_text("", encoding="utf-8")
            with patch.dict(os.environ, {"HOME": tmp_dir}, clear=False):
                with patch("autoquant_cli.readme_sync.sync_setup_repo", return_value=sync_result):
                    with patch(
                        "autoquant_cli.readme_sync.subprocess.run",
                        return_value=subprocess.CompletedProcess(["pip"], 0, "ok", ""),
                    ):
                        result = run_update()
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["setup_repo_dir"], "/tmp/autoquant-setup")
        self.assertEqual(result["setup_repo_head_after_update"], "new_head")
        self.assertTrue(result["setup_repo_updated"])

    def test_run_update_raises_when_pip_fails(self) -> None:
        sync_result = {
            "repo_dir": "/tmp/autoquant-setup",
            "repo_url": "https://github.com/autoquantai/autoquant-setup.git",
            "branch": "main",
            "baseline_commit": "old_head",
            "latest_commit": "new_head",
            "head_after_sync": "new_head",
            "updated": True,
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            pip_path = Path(tmp_dir) / ".nanobot" / "workspace" / "autoquant" / "venv" / "autoquant" / "bin" / "pip"
            pip_path.parent.mkdir(parents=True, exist_ok=True)
            pip_path.write_text("", encoding="utf-8")
            with patch.dict(os.environ, {"HOME": tmp_dir}, clear=False):
                with patch("autoquant_cli.readme_sync.sync_setup_repo", return_value=sync_result):
                    with patch(
                        "autoquant_cli.readme_sync.subprocess.run",
                        return_value=subprocess.CompletedProcess(["pip"], 1, "out", "err"),
                    ):
                        with self.assertRaises(RuntimeError):
                            run_update()

    def test_run_update_uses_autoquant_workspace_override(self) -> None:
        sync_result = {
            "repo_dir": "/tmp/autoquant-setup",
            "repo_url": "https://github.com/autoquantai/autoquant-setup.git",
            "branch": "main",
            "baseline_commit": "old_head",
            "latest_commit": "new_head",
            "head_after_sync": "new_head",
            "updated": True,
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_dir = Path(tmp_dir) / "custom-workspace"
            pip_path = workspace_dir / "venv" / "autoquant" / "bin" / "pip"
            pip_path.parent.mkdir(parents=True, exist_ok=True)
            pip_path.write_text("", encoding="utf-8")
            with patch.dict(
                os.environ,
                {"HOME": tmp_dir, "AUTOQUANT_WORKSPACE": str(workspace_dir)},
                clear=False,
            ):
                with patch("autoquant_cli.readme_sync.sync_setup_repo", return_value=sync_result):
                    with patch(
                        "autoquant_cli.readme_sync.subprocess.run",
                        return_value=subprocess.CompletedProcess(["pip"], 0, "ok", ""),
                    ):
                        result = run_update()
        self.assertEqual(result["exit_code"], 0)

    def test_get_readme_diff_returns_expected_payload(self) -> None:
        sync_state = {
            "repo_dir": "/tmp/autoquant-setup",
            "repo_url": "https://github.com/autoquantai/autoquant-setup.git",
            "branch": "main",
            "baseline_commit": "abc",
            "latest_commit": "def",
        }

        def fake_diff(repo_dir: Path, baseline: str, latest: str, rel_path: str) -> str:
            if rel_path == "README.md":
                return "--- a/README.md\n+++ b/README.md\n"
            return ""

        with patch("autoquant_cli.readme_sync.ensure_setup_repo_synced", return_value=sync_state):
            with patch("autoquant_cli.readme_sync.diff_file_between_refs", side_effect=fake_diff):
                result = get_readme_diff()
        self.assertEqual(result["baseline_commit"], "abc")
        self.assertEqual(result["latest_commit"], "def")
        self.assertTrue(result["has_changes"])
        self.assertEqual(len(result["files"]), 3)
        changed_paths = [item["path"] for item in result["files"] if item["changed"]]
        self.assertEqual(changed_paths, ["README.md"])


if __name__ == "__main__":
    unittest.main()

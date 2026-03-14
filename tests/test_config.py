from __future__ import annotations

import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class ConfigTest(unittest.TestCase):
    def test_load_env_uses_default_workspace_env_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            home_path = Path(tmp_dir)
            workspace_dir = home_path / ".nanobot" / "workspace" / "autoquant"
            workspace_dir.mkdir(parents=True, exist_ok=True)
            (workspace_dir / ".env").write_text("AUTOQUANT_API_URL=https://home-env.example\n", encoding="utf-8")
            (home_path / ".env").write_text("AUTOQUANT_API_URL=https://wrong.example\n", encoding="utf-8")
            with patch.dict(os.environ, {"HOME": tmp_dir}, clear=False):
                os.environ.pop("AUTOQUANT_API_URL", None)
                os.environ.pop("AUTOQUANT_WORKSPACE", None)
                import autoquant_cli.config as config

                importlib.reload(config)

                self.assertEqual(config.get_workspace_root(), workspace_dir)
                self.assertEqual(config.ENV_FILE_PATH, workspace_dir / ".env")
                self.assertEqual(config.get_backend_base_url(), "https://home-env.example")

    def test_load_env_uses_autoquant_workspace_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            home_path = Path(tmp_dir)
            workspace_dir = home_path / "custom-workspace"
            workspace_dir.mkdir(parents=True, exist_ok=True)
            (workspace_dir / ".env").write_text("AUTOQUANT_API_URL=https://override.example\n", encoding="utf-8")
            with patch.dict(
                os.environ,
                {"HOME": tmp_dir, "AUTOQUANT_WORKSPACE": str(workspace_dir)},
                clear=False,
            ):
                os.environ.pop("AUTOQUANT_API_URL", None)
                import autoquant_cli.config as config

                importlib.reload(config)

                self.assertEqual(config.get_workspace_root(), workspace_dir)
                self.assertEqual(config.ENV_FILE_PATH, workspace_dir / ".env")
                self.assertEqual(config.get_backend_base_url(), "https://override.example")

    def test_backend_url_requires_autoquant_api_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.dict(os.environ, {"HOME": tmp_dir}, clear=True):
                import autoquant_cli.config as config

                importlib.reload(config)

                with self.assertRaises(RuntimeError) as context:
                    config.get_backend_base_url()
                self.assertIn("AUTOQUANT_API_URL", str(context.exception))


if __name__ == "__main__":
    unittest.main()

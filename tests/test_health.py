from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import URLError

from autoquant_cli.commands.health import health


class HealthTest(unittest.TestCase):
    def test_health_reports_success(self) -> None:
        response = MagicMock()
        response.read.return_value = b'{"message":"pong"}'
        response.__enter__.return_value = response
        response.__exit__.return_value = None
        with (
            patch("autoquant_cli.commands.health.get_env", side_effect=lambda name, required=False: {
                "MASSIVE_API_KEY": "massive",
                "AUTOQUANT_API_KEY": "api-key",
                "AUTOQUANT_API_URL": "http://127.0.0.1:8000",
            }.get(name)),
            patch("autoquant_cli.commands.health.urlopen", return_value=response),
            patch("autoquant_cli.commands.health.ENV_FILE_PATH", Path("/tmp/test.env")),
            patch("pathlib.Path.exists", return_value=True),
        ):
            payload = health()
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["backend"]["ok"])
        self.assertTrue(payload["env"]["AUTOQUANT_API_KEY"]["present"])

    def test_health_reports_backend_failure(self) -> None:
        with (
            patch("autoquant_cli.commands.health.get_env", side_effect=lambda name, required=False: {
                "MASSIVE_API_KEY": "massive",
                "AUTOQUANT_API_KEY": "api-key",
                "AUTOQUANT_API_URL": "http://127.0.0.1:8000",
            }.get(name)),
            patch("autoquant_cli.commands.health.urlopen", side_effect=URLError("connection refused")),
            patch("autoquant_cli.commands.health.ENV_FILE_PATH", Path("/tmp/test.env")),
            patch("pathlib.Path.exists", return_value=True),
        ):
            payload = health()
        self.assertFalse(payload["ok"])
        self.assertFalse(payload["backend"]["ok"])
        self.assertIn("connection refused", payload["backend"]["error"])


if __name__ == "__main__":
    unittest.main()

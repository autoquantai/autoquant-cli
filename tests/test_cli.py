from __future__ import annotations

import unittest
from unittest.mock import patch

from smartpy.utility.log_util import getLogger
from typer.testing import CliRunner

from autoquant_cli.cli import app

logger = getLogger(__name__)


class CliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_help_lists_commands(self) -> None:
        logger.info("Testing CLI help output")
        result = self.runner.invoke(app, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("create-experiment", result.stdout)
        self.assertIn("validate-model", result.stdout)
        self.assertIn("run-model", result.stdout)
        self.assertIn("api", result.stdout)
        self.assertIn("health", result.stdout)
        self.assertIn("get-openapi", result.stdout)
        self.assertIn("get-readme-diff", result.stdout)
        self.assertIn("run-update", result.stdout)

    def test_api_command_rejects_invalid_json(self) -> None:
        result = self.runner.invoke(app, ["api", "/experiment/get", "{bad"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Invalid JSON payload", result.output)

    def test_api_command_prints_json(self) -> None:
        with patch("autoquant_cli.cli.post_json", return_value={"ok": True}):
            result = self.runner.invoke(app, ["api", "/experiment/get", '{"run_id":"run-1"}'])
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.output.strip(), '{"ok": true}')

    def test_health_command_prints_json(self) -> None:
        with patch("autoquant_cli.cli.health", return_value={"ok": True}):
            result = self.runner.invoke(app, ["health"])
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.output.strip(), '{"ok": true}')

    def test_get_openapi_command_prints_pretty_json(self) -> None:
        payload = {"openapi": "3.1.0", "info": {"title": "AutoQuant API"}}
        with patch("autoquant_cli.cli.get_openapi_json", return_value=payload):
            result = self.runner.invoke(app, ["get-openapi"])
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.output, '{\n  "openapi": "3.1.0",\n  "info": {\n    "title": "AutoQuant API"\n  }\n}\n')

    def test_get_readme_diff_command_prints_json(self) -> None:
        with patch("autoquant_cli.cli.get_readme_diff", return_value={"has_changes": True}):
            result = self.runner.invoke(app, ["get-readme-diff"])
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.output.strip(), '{"has_changes": true}')

    def test_run_update_command_prints_json(self) -> None:
        with patch("autoquant_cli.cli.run_update", return_value={"exit_code": 0}):
            result = self.runner.invoke(app, ["run-update"])
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.output.strip(), '{"exit_code": 0}')

    def test_create_experiment_command_prints_json(self) -> None:
        with patch("autoquant_cli.cli.create_experiment", return_value={"id": "run-1", "data_source": "downloaded"}):
            result = self.runner.invoke(
                app,
                [
                    "create-experiment",
                    "--name",
                    "test-run",
                    "--ticker",
                    "AAPL",
                    "--from-date",
                    "2026-01-01",
                    "--to-date",
                    "2026-02-28",
                    "--task",
                    "classification",
                ],
            )
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.output.strip(), '{"id": "run-1", "data_source": "downloaded"}')


if __name__ == "__main__":
    unittest.main()

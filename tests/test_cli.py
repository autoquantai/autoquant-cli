from __future__ import annotations

import logging
import unittest
from unittest.mock import patch

from typer.testing import CliRunner

from autoquant_cli.cli import app

logger = logging.getLogger(__name__)


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

    def test_create_experiment_command_prints_json(self) -> None:
        with patch("autoquant_cli.cli.create_experiment", return_value={"id": "run-1", "data_source": "downloaded"}):
            result = self.runner.invoke(
                app,
                [
                    "create-experiment",
                    "--name",
                    "test-run",
                    "--input-ohlc-tickers",
                    "MSFT",
                    "--input-ohlc-tickers",
                    "QQQ",
                    "--target-ticker",
                    "AAPL",
                    "--data-provider",
                    "ccxt",
                    "--ccxt-exchange",
                    "kraken",
                    "--from-date",
                    "2025-01-01",
                    "--to-date",
                    "2026-01-01",
                    "--task",
                    "classification",
                ],
            )
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.output.strip(), '{"id": "run-1", "data_source": "downloaded"}')

    def test_create_experiment_command_rejects_short_window(self) -> None:
        result = self.runner.invoke(
            app,
            [
                "create-experiment",
                "--name",
                "test-run",
                "--target-ticker",
                "AAPL",
                "--from-date",
                "2026-02-10",
                "--to-date",
                "2026-08-20",
                "--task",
                "classification",
            ],
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIsInstance(result.exception, RuntimeError)
        self.assertIn("Experiment requires at least 365 days of data", str(result.exception))


if __name__ == "__main__":
    unittest.main()

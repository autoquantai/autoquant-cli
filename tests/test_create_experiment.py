from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autoquant_cli.commands.create_experiment import create_experiment

logger = logging.getLogger(__name__)


class CreateExperimentTest(unittest.TestCase):
    def test_create_experiment_posts_run_and_sets_up_local_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_root = Path(tmp_dir) / "runs" / "run-1"
            calls: list[tuple[str, dict]] = []

            def fake_post_json(path: str, payload: dict) -> dict:
                calls.append((path, payload))
                return {
                    "id": "run-1",
                    "name": payload["name"],
                    "input_ohlc_tickers": payload["input_ohlc_tickers"],
                    "target_ticker": payload["target_ticker"],
                    "data_provider": payload["data_provider"],
                    "ccxt_exchange": payload["ccxt_exchange"],
                    "from_date": payload["from_date"],
                    "to_date": payload["to_date"],
                    "task": payload["task"],
                    "train_time_limit_minutes": payload["train_time_limit_minutes"],
                    "max_experiments": payload["max_experiments"],
                    "status": "pending",
                }

            with (
                patch("autoquant_cli.commands.create_experiment.post_json", side_effect=fake_post_json),
                patch("autoquant_cli.commands.create_experiment.run_dir", return_value=run_root),
                patch("autoquant_cli.commands.create_experiment.ensure_run_prices", return_value="downloaded") as ensure_run_prices_mock,
            ):
                logger.info("Testing create-experiment payload mapping")
                result = create_experiment(
                    name="test-run",
                    input_ohlc_tickers=["MSFT", "QQQ"],
                    target_ticker="AAPL",
                    data_provider="ccxt",
                    ccxt_exchange="kraken",
                    from_date="2025-01-01",
                    to_date="2026-01-01",
                    task="classification",
                    max_experiments=12,
                    train_time_limit_minutes=45,
                )

            self.assertEqual(len(calls), 1)
            path, payload = calls[0]
            self.assertEqual(path, "/run/create")
            self.assertEqual(
                payload,
                {
                    "name": "test-run",
                    "input_ohlc_tickers": ["MSFT", "QQQ"],
                    "target_ticker": "AAPL",
                    "data_provider": "ccxt",
                    "ccxt_exchange": "kraken",
                    "from_date": "2025-01-01",
                    "to_date": "2026-01-01",
                    "task": "classification",
                    "max_experiments": 12,
                    "train_time_limit_minutes": 45,
                },
            )
            ensure_run_prices_mock.assert_called_once_with(
                "run-1",
                ["MSFT", "QQQ"],
                "AAPL",
                "2025-01-01",
                "2026-01-01",
                data_provider="ccxt",
                ccxt_exchange="kraken",
                force_refresh=False,
            )
            self.assertEqual(result["id"], "run-1")
            self.assertEqual(result["run_dir"], str(run_root))
            self.assertEqual(result["data_source"], "downloaded")

    def test_create_experiment_rejects_short_window_before_side_effects(self) -> None:
        with (
            patch("autoquant_cli.commands.create_experiment.post_json") as post_json_mock,
            patch("autoquant_cli.commands.create_experiment.ensure_run_prices") as ensure_run_prices_mock,
        ):
            with self.assertRaises(RuntimeError) as context:
                create_experiment(
                    name="test-run",
                    input_ohlc_tickers=[],
                    target_ticker="AAPL",
                    data_provider="massive",
                    ccxt_exchange=None,
                    from_date="2026-02-10",
                    to_date="2026-08-20",
                    task="classification",
                    max_experiments=12,
                    train_time_limit_minutes=45,
                )
        self.assertIn("Experiment requires at least 365 days of data", str(context.exception))
        post_json_mock.assert_not_called()
        ensure_run_prices_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()

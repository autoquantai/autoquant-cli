from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autoquant_cli.commands.validate_model import validate_model

logger = logging.getLogger(__name__)


class ValidateModelTest(unittest.TestCase):
    def test_validate_model_returns_completed_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_file = Path(tmp_dir) / "model.py"
            model_file.write_text("class Placeholder:\n    pass\n", encoding="utf-8")
            logger.info("Testing validate-model success mapping")
            ensure_run_prices_mock = None
            run_train_file_mock = None
            with (
                patch(
                    "autoquant_cli.commands.validate_model.get_run",
                    return_value={
                        "id": "run-1",
                        "input_ohlc_tickers": ["MSFT"],
                        "target_ticker": "AAPL",
                        "data_provider": "ccxt",
                        "ccxt_exchange": "binance",
                        "from_date": "2026-01-01",
                        "to_date": "2026-02-28",
                        "task": "classification",
                    },
                ),
                patch("autoquant_cli.commands.validate_model.read_csv", return_value=[]),
                patch("autoquant_cli.commands.validate_model.ensure_run_prices", return_value="downloaded") as ensure_run_prices_mock,
                patch(
                    "autoquant_cli.commands.validate_model.run_train_file",
                    return_value={
                        "train": {
                            "weighted avg": {"f1-score": 0.8},
                            "selected_hyperparams": {"training_size_days": 18},
                        },
                        "validation": {
                            "0": {"precision": 1.0, "recall": 1.0, "f1-score": 1.0, "support": 1.0},
                            "1": {"precision": 0.5, "recall": 1.0, "f1-score": 2 / 3, "support": 1.0},
                            "accuracy": 0.75,
                            "macro avg": {"precision": 0.75, "recall": 1.0, "f1-score": 5 / 6, "support": 2.0},
                            "weighted avg": {"precision": 0.75, "recall": 0.75, "f1-score": 0.7, "support": 2.0},
                        },
                        "stdout": "",
                        "stderr": "",
                    },
                ) as run_train_file_mock,
            ):
                payload = validate_model(str(model_file), "run-1", "classification", refresh_data=False)
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["metrics"]["weighted avg"]["f1-score"], 0.7)
            self.assertIsNone(payload["error"])
            self.assertEqual(payload["data_source"], "downloaded")
            self.assertTrue(payload["sandbox_initialized"])
            self.assertEqual(payload["training_size_days"], 18)
            self.assertEqual(payload["test_size_days"], 7)
            self.assertEqual(payload["validation_run_id"], "run-1")
            self.assertEqual(payload["input_ohlc_tickers"], ["MSFT"])
            self.assertEqual(payload["target_ticker"], "AAPL")
            self.assertEqual(payload["data_provider"], "ccxt")
            ensure_run_prices_mock.assert_called_once_with(
                "run-1",
                ["MSFT"],
                "AAPL",
                "2026-01-01",
                "2026-02-28",
                data_provider="ccxt",
                ccxt_exchange="binance",
                force_refresh=False,
            )
            run_train_file_mock.assert_called_once_with(
                model_file,
                run_id="run-1",
                model_id="model",
                expected_task="classification",
                training_size_days=15,
                execution_profile="sandbox",
            )

    def test_validate_model_returns_failed_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_file = Path(tmp_dir) / "model.py"
            model_file.write_text("class Placeholder:\n    pass\n", encoding="utf-8")
            ensure_run_prices_mock = None
            run_train_file_mock = None
            with (
                patch(
                    "autoquant_cli.commands.validate_model.get_run",
                    return_value={
                        "id": "run-2",
                        "input_ohlc_tickers": [],
                        "target_ticker": "BTC/USDT",
                        "data_provider": "ccxt",
                        "ccxt_exchange": "binance",
                        "from_date": "2026-01-01",
                        "to_date": "2026-02-28",
                        "task": "classification",
                    },
                ),
                patch("autoquant_cli.commands.validate_model.read_csv", return_value=[{"timestamp": "1"}]),
                patch("autoquant_cli.commands.validate_model.ensure_run_prices", return_value="reused") as ensure_run_prices_mock,
                patch(
                    "autoquant_cli.commands.validate_model.run_train_file",
                    return_value={"stdout": "printed", "stderr": "bad stderr", "runtime_error": "boom"},
                ) as run_train_file_mock,
            ):
                payload = validate_model(str(model_file), "run-2", "classification", refresh_data=False)
            self.assertEqual(payload["status"], "failed")
            self.assertIsNone(payload["metrics"])
            self.assertIn("boom", payload["error"])
            self.assertFalse(payload["sandbox_initialized"])
            ensure_run_prices_mock.assert_called_once_with(
                "run-2",
                [],
                "BTC/USDT",
                "2026-01-01",
                "2026-02-28",
                data_provider="ccxt",
                ccxt_exchange="binance",
                force_refresh=False,
            )
            run_train_file_mock.assert_called_once_with(
                model_file,
                run_id="run-2",
                model_id="model",
                expected_task="classification",
                training_size_days=15,
                execution_profile="sandbox",
            )


if __name__ == "__main__":
    unittest.main()

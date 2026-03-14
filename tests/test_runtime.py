from __future__ import annotations

import csv
import json
import logging
import os
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from autoquant_cli.quant.runtime import run_train_file

logger = logging.getLogger(__name__)


def write_prices(prices_file: Path, hours: int) -> None:
    prices_file.parent.mkdir(parents=True, exist_ok=True)
    with prices_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["timestamp", "aapl_open", "aapl_high", "aapl_low", "aapl_close", "aapl_volume"],
        )
        writer.writeheader()
        started_at = datetime(2026, 1, 1, tzinfo=UTC)
        for index in range(hours):
            writer.writerow(
                {
                    "timestamp": (started_at + timedelta(hours=index)).isoformat(),
                    "aapl_open": "1",
                    "aapl_high": "2",
                    "aapl_low": "0",
                    "aapl_close": str(index % 2),
                    "aapl_volume": "100",
                }
            )


def write_run_metadata(home_path: Path, run_id: str, target_ticker: str = "AAPL", input_ohlc_tickers: list[str] | None = None) -> None:
    metadata_file = home_path / ".nanobot" / "workspace" / "autoquant" / "runs" / run_id / "data" / "run_metadata.json"
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    metadata_file.write_text(
        json.dumps(
            {
                "input_ohlc_tickers": input_ohlc_tickers or [],
                "target_ticker": target_ticker,
                "data_provider": "massive",
                "ccxt_exchange": None,
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )


class RuntimeTest(unittest.TestCase):
    def test_run_train_file_supports_quant_model_base_import_in_sandbox(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            home_path = Path(tmp_dir)
            prices_file = home_path / ".nanobot" / "workspace" / "autoquant" / "runs" / "run-1" / "data" / "prices.csv"
            write_prices(prices_file, 72)
            write_run_metadata(home_path, "run-1")
            model_file = Path(tmp_dir) / "model.py"
            model_file.write_text(
                "\n".join(
                    [
                        "from autoquant_cli.quant.model_base import AutoQuantModel",
                        "",
                        "class ExampleModel(AutoQuantModel):",
                        "    def create_features(self, frame):",
                        "        frame = frame.copy()",
                        "        frame['feature'] = frame['close'].astype(float)",
                        "        frame['target'] = frame['close'].astype(int)",
                        "        return frame, ['feature']",
                        "",
                        "    def fit(self, x_train, y_train, hyperparams):",
                        "        self.threshold = 0.5",
                        "",
                        "    def predict(self, x_test):",
                        "        return [1 if value > self.threshold else 0 for value in x_test['feature'].tolist()]",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            logger.info("Running runtime compatibility test")
            with patch.dict(
                os.environ,
                {"HOME": tmp_dir, "AUTOQUANT_WORKSPACE": str(home_path / ".nanobot" / "workspace" / "autoquant")},
                clear=False,
            ):
                result = run_train_file(
                    model_file,
                    run_id="run-1",
                    expected_task="classification",
                    execution_profile="sandbox",
                )
            self.assertIn("train", result)
            self.assertIn("validation", result)
            self.assertNotIn("runtime_error", result)
            self.assertIn("selected_hyperparams", result["train"])
            self.assertEqual(int(result["train"]["selected_hyperparams"]["training_size_days"]), 3)

    def test_run_train_file_rejects_short_default_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            home_path = Path(tmp_dir)
            prices_file = home_path / ".nanobot" / "workspace" / "autoquant" / "runs" / "run-2" / "data" / "prices.csv"
            write_prices(prices_file, 72)
            write_run_metadata(home_path, "run-2")
            model_file = Path(tmp_dir) / "model.py"
            model_file.write_text(
                "\n".join(
                    [
                        "from autoquant_cli.quant.model_base import AutoQuantModel",
                        "",
                        "class ExampleModel(AutoQuantModel):",
                        "    def create_features(self, frame):",
                        "        frame = frame.copy()",
                        "        frame['feature'] = frame['close'].astype(float)",
                        "        frame['target'] = frame['close'].astype(int)",
                        "        return frame, ['feature']",
                        "",
                        "    def fit(self, x_train, y_train, hyperparams):",
                        "        self.threshold = 0.5",
                        "",
                        "    def predict(self, x_test):",
                        "        return [1 if value > self.threshold else 0 for value in x_test['feature'].tolist()]",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"HOME": tmp_dir, "AUTOQUANT_WORKSPACE": str(home_path / ".nanobot" / "workspace" / "autoquant")},
                clear=False,
            ):
                result = run_train_file(model_file, run_id="run-2", expected_task="classification")
            self.assertIn("runtime_error", result)
            self.assertIn("Need at least 220 OHLCV rows", result["runtime_error"])

    def test_run_train_file_samples_dict_based_hyperparameters_in_sandbox(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            home_path = Path(tmp_dir)
            prices_file = home_path / ".nanobot" / "workspace" / "autoquant" / "runs" / "run-3" / "data" / "prices.csv"
            write_prices(prices_file, 72)
            write_run_metadata(home_path, "run-3")
            model_file = Path(tmp_dir) / "search_space_model.py"
            model_file.write_text(
                "\n".join(
                    [
                        "from autoquant_cli.quant.model_base import AutoQuantModel",
                        "",
                        "class ExampleModel(AutoQuantModel):",
                        "    def create_features(self, frame):",
                        "        frame = frame.copy()",
                        "        frame['feature'] = frame['close'].astype(float)",
                        "        frame['target'] = frame['close'].astype(int)",
                        "        return frame, ['feature']",
                        "",
                        "    def get_hyperparameter_candidates(self):",
                        "        return {'threshold': [0.25, 0.5, 0.75], 'bias': (-1.0, 1.0)}",
                        "",
                        "    def fit(self, x_train, y_train, hyperparams):",
                        "        self.threshold = float(hyperparams['threshold']) + float(hyperparams['bias'])",
                        "",
                        "    def predict(self, x_test):",
                        "        return [1 if value > self.threshold else 0 for value in x_test['feature'].tolist()]",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"HOME": tmp_dir, "AUTOQUANT_WORKSPACE": str(home_path / ".nanobot" / "workspace" / "autoquant")},
                clear=False,
            ):
                result = run_train_file(
                    model_file,
                    run_id="run-3",
                    expected_task="classification",
                    execution_profile="sandbox",
                )
            self.assertNotIn("runtime_error", result)
            self.assertIn("train", result)
            self.assertIn("selected_hyperparams", result["train"])
            self.assertIn("threshold", result["train"]["selected_hyperparams"])
            self.assertIn("bias", result["train"]["selected_hyperparams"])
            self.assertEqual(int(result["train"]["selected_hyperparams"]["training_size_days"]), 3)
            self.assertEqual(int(result["train"]["hyperparam_candidates_attempted"]), 1)


if __name__ == "__main__":
    unittest.main()

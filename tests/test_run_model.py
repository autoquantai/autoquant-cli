from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autoquant_cli.commands.run_model import run_model

logger = logging.getLogger(__name__)


class RunModelTest(unittest.TestCase):
    def test_run_model_posts_expected_experiment_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_file = Path(tmp_dir) / "model.py"
            model_file.write_text("print('x')\n", encoding="utf-8")
            calls: list[tuple[str, dict]] = []

            def fake_post_json(path: str, payload: dict) -> dict:
                calls.append((path, payload))
                return {"id": "exp-1", "run_id": payload["run_id"]}

            ensure_run_prices_mock = None
            with (
                patch(
                    "autoquant_cli.commands.run_model.get_run",
                    return_value={
                        "id": "run-1",
                        "input_ohlc_tickers": ["MSFT"],
                        "target_ticker": "AAPL",
                        "data_provider": "ccxt",
                        "ccxt_exchange": "binance",
                        "from_date": "2026-01-01",
                        "to_date": "2026-02-28",
                        "task": "classification",
                        "train_time_limit_minutes": 12,
                    },
                ),
                patch("autoquant_cli.commands.run_model.ensure_run_prices", return_value="reused") as ensure_run_prices_mock,
                patch(
                    "autoquant_cli.commands.run_model.run_train_file",
                    return_value={
                        "train": {
                            "weighted avg": {"f1-score": 0.9},
                            "selected_hyperparams": {"training_size_days": 21, "max_depth": 4},
                        },
                        "validation": {
                            "0": {"precision": 1.0, "recall": 1.0, "f1-score": 1.0, "support": 1.0},
                            "1": {"precision": 0.6, "recall": 1.0, "f1-score": 0.75, "support": 1.0},
                            "accuracy": 0.8,
                            "macro avg": {"precision": 0.8, "recall": 1.0, "f1-score": 0.875, "support": 2.0},
                            "weighted avg": {"precision": 0.8, "recall": 0.8, "f1-score": 0.8, "support": 2.0},
                        },
                        "stdout": "ok",
                        "stderr": "",
                    },
                ),
                patch("autoquant_cli.commands.run_model.post_json", side_effect=fake_post_json),
            ):
                logger.info("Testing run-model payload mapping")
                result = run_model(
                    run_id="run-1",
                    name="candidate-1",
                    generation=2,
                    model_path=str(model_file),
                    log="test log",
                    parent_ids=["parent-1"],
                    reasoning="why",
                    task=None,
                )

            self.assertEqual(result["id"], "exp-1")
            self.assertEqual(result["data_source"], "reused")
            self.assertEqual(len(calls), 1)
            path, payload = calls[0]
            self.assertEqual(path, "/experiment/create")
            self.assertEqual(payload["run_id"], "run-1")
            self.assertEqual(payload["name"], "candidate-1")
            self.assertEqual(payload["generation"], 2)
            self.assertEqual(payload["parent_ids"], ["parent-1"])
            self.assertEqual(payload["reasoning"], "why")
            self.assertEqual(payload["error"], None)
            self.assertIn("source", payload["model"])
            self.assertEqual(payload["model"]["hyperparameters"], {"training_size_days": 21, "max_depth": 4})
            self.assertEqual(payload["evals"]["validation"]["weighted avg"]["f1-score"], 0.8)
            self.assertEqual(payload["evals"]["stdout"], "ok")
            ensure_run_prices_mock.assert_called_once_with(
                "run-1",
                ["MSFT"],
                "AAPL",
                "2026-01-01",
                "2026-02-28",
                data_provider="ccxt",
                ccxt_exchange="binance",
            )

    def test_run_model_posts_null_parent_ids_for_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_file = Path(tmp_dir) / "model.py"
            model_file.write_text("print('x')\n", encoding="utf-8")
            calls: list[tuple[str, dict]] = []

            def fake_post_json(path: str, payload: dict) -> dict:
                calls.append((path, payload))
                return {"id": "exp-root", "run_id": payload["run_id"]}

            with (
                patch(
                    "autoquant_cli.commands.run_model.get_run",
                    return_value={
                        "id": "run-1",
                        "input_ohlc_tickers": [],
                        "target_ticker": "AAPL",
                        "data_provider": "massive",
                        "ccxt_exchange": None,
                        "from_date": "2026-01-01",
                        "to_date": "2026-02-28",
                        "task": "classification",
                        "train_time_limit_minutes": 12,
                    },
                ),
                patch("autoquant_cli.commands.run_model.ensure_run_prices", return_value="reused"),
                patch(
                    "autoquant_cli.commands.run_model.run_train_file",
                    return_value={
                        "train": {"selected_hyperparams": {"training_size_days": 21}},
                        "validation": {"weighted avg": {"f1-score": 0.8}},
                        "stdout": "ok",
                        "stderr": "",
                    },
                ),
                patch("autoquant_cli.commands.run_model.post_json", side_effect=fake_post_json),
            ):
                run_model(
                    run_id="run-1",
                    name="root-candidate",
                    generation=0,
                    model_path=str(model_file),
                    log="root",
                )

            _, payload = calls[0]
            self.assertIsNone(payload["parent_ids"])


if __name__ == "__main__":
    unittest.main()

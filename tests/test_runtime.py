from __future__ import annotations

import csv
import os
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from smartpy.utility.log_util import getLogger

from autoquant_cli.runtime import run_train_file

logger = getLogger(__name__)


class RuntimeTest(unittest.TestCase):
    def test_run_train_file_supports_core_model_base_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            home_path = Path(tmp_dir)
            prices_file = home_path / ".autoquant" / "runs" / "run-1" / "data" / "prices.csv"
            prices_file.parent.mkdir(parents=True, exist_ok=True)
            with prices_file.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=["timestamp", "ticker", "open", "high", "low", "close", "volume"])
                writer.writeheader()
                started_at = datetime(2026, 1, 1, tzinfo=UTC)
                for index in range(2200):
                    writer.writerow(
                        {
                            "timestamp": (started_at + timedelta(hours=index)).isoformat(),
                            "ticker": "AAPL",
                            "open": "1",
                            "high": "2",
                            "low": "0",
                            "close": str(index % 2),
                            "volume": "100",
                        }
                    )
            model_file = Path(tmp_dir) / "model.py"
            model_file.write_text(
                "\n".join(
                    [
                        "from core.model_base import AutoQuantModel",
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
            with patch.dict(os.environ, {"HOME": tmp_dir}, clear=False):
                result = run_train_file(model_file, run_id="run-1", expected_task="classification")
            self.assertIn("train", result)
            self.assertIn("validation", result)
            self.assertNotIn("runtime_error", result)


if __name__ == "__main__":
    unittest.main()

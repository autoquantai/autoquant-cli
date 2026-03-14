from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from smartpy.utility.log_util import getLogger

from autoquant_cli.validate_model import validate_model

logger = getLogger(__name__)


class ValidateModelTest(unittest.TestCase):
    def test_validate_model_returns_completed_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_file = Path(tmp_dir) / "model.py"
            model_file.write_text("class Placeholder:\n    pass\n", encoding="utf-8")
            logger.info("Testing validate-model success mapping")
            with (
                patch("autoquant_cli.validate_model.read_csv", return_value=[]),
                patch("autoquant_cli.validate_model.ensure_run_prices", return_value="downloaded"),
                patch(
                    "autoquant_cli.validate_model.run_train_file",
                    return_value={"train": {"weighted_f1": 0.8}, "validation": {"weighted_f1": 0.7}, "stdout": "", "stderr": ""},
                ),
            ):
                payload = validate_model(str(model_file), "classification", refresh_data=False)
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["metrics"], {"weighted_f1": 0.7})
            self.assertIsNone(payload["error"])
            self.assertEqual(payload["data_source"], "downloaded")
            self.assertTrue(payload["sandbox_initialized"])

    def test_validate_model_returns_failed_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_file = Path(tmp_dir) / "model.py"
            model_file.write_text("class Placeholder:\n    pass\n", encoding="utf-8")
            with (
                patch("autoquant_cli.validate_model.read_csv", return_value=[{"timestamp": "1"}]),
                patch("autoquant_cli.validate_model.ensure_run_prices", return_value="reused"),
                patch(
                    "autoquant_cli.validate_model.run_train_file",
                    return_value={"stdout": "printed", "stderr": "bad stderr", "runtime_error": "boom"},
                ),
            ):
                payload = validate_model(str(model_file), "classification", refresh_data=False)
            self.assertEqual(payload["status"], "failed")
            self.assertIsNone(payload["metrics"])
            self.assertIn("boom", payload["error"])
            self.assertFalse(payload["sandbox_initialized"])


if __name__ == "__main__":
    unittest.main()

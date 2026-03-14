from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autoquant_cli.run_model import run_model

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

            with (
                patch(
                    "autoquant_cli.run_model.get_run",
                    return_value={
                        "id": "run-1",
                        "ticker": "AAPL",
                        "from_date": "2026-01-01",
                        "to_date": "2026-02-28",
                        "task": "classification",
                        "train_time_limit_minutes": 12,
                    },
                ),
                patch("autoquant_cli.run_model.ensure_run_prices", return_value="reused"),
                patch(
                    "autoquant_cli.run_model.run_train_file",
                    return_value={
                        "train": {"weighted_f1": 0.9},
                        "validation": {"weighted_f1": 0.8},
                        "stdout": "ok",
                        "stderr": "",
                    },
                ),
                patch("autoquant_cli.run_model.post_json", side_effect=fake_post_json),
            ):
                logger.info("Testing run-model payload mapping")
                result = run_model(
                    run_id="run-1",
                    name="candidate-1",
                    generation=2,
                    model_path=str(model_file),
                    log="test log",
                    parent_id="parent-1",
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
            self.assertEqual(payload["parent_id"], "parent-1")
            self.assertEqual(payload["reasoning"], "why")
            self.assertEqual(payload["error"], None)
            self.assertIn("source", payload["model"])
            self.assertEqual(payload["evals"]["validation"], {"weighted_f1": 0.8})
            self.assertEqual(payload["evals"]["stdout"], "ok")


if __name__ == "__main__":
    unittest.main()

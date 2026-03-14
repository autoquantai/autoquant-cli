from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autoquant_cli.create_experiment import create_experiment

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
                    "ticker": payload["ticker"],
                    "from_date": payload["from_date"],
                    "to_date": payload["to_date"],
                    "task": payload["task"],
                    "train_time_limit_minutes": payload["train_time_limit_minutes"],
                    "max_experiments": payload["max_experiments"],
                    "status": "pending",
                }

            with (
                patch("autoquant_cli.create_experiment.post_json", side_effect=fake_post_json),
                patch("autoquant_cli.create_experiment.run_dir", return_value=run_root),
                patch("autoquant_cli.create_experiment.ensure_run_prices", return_value="downloaded") as ensure_run_prices_mock,
            ):
                logger.info("Testing create-experiment payload mapping")
                result = create_experiment(
                    name="test-run",
                    ticker="AAPL",
                    from_date="2026-01-01",
                    to_date="2026-02-28",
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
                    "ticker": "AAPL",
                    "from_date": "2026-01-01",
                    "to_date": "2026-02-28",
                    "task": "classification",
                    "max_experiments": 12,
                    "train_time_limit_minutes": 45,
                },
            )
            ensure_run_prices_mock.assert_called_once_with(
                "run-1",
                "AAPL",
                "2026-01-01",
                "2026-02-28",
                force_refresh=False,
            )
            self.assertEqual(result["id"], "run-1")
            self.assertEqual(result["run_dir"], str(run_root))
            self.assertEqual(result["fetch_from_date"], "2025-12-02")
            self.assertEqual(result["data_source"], "downloaded")


if __name__ == "__main__":
    unittest.main()

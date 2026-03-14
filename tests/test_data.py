from __future__ import annotations

import os
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from autoquant_cli.quant.data import ensure_run_prices, load_dataset, prices_path, raw_prices_path, read_csv


class DataTest(unittest.TestCase):
    def test_ensure_run_prices_writes_merged_prices_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            home_path = Path(tmp_dir)
            started_at = datetime(2026, 1, 1, tzinfo=UTC)
            target_rows = []
            input_rows = []
            for index in range(5):
                target_ts = started_at + timedelta(hours=index)
                input_ts = target_ts + timedelta(minutes=5)
                target_rows.append(
                    {
                        "timestamp": target_ts.isoformat(),
                        "ticker": "AAPL",
                        "open": "1",
                        "high": "2",
                        "low": "0",
                        "close": str(index),
                        "volume": "100",
                    }
                )
                input_rows.append(
                    {
                        "timestamp": input_ts.isoformat(),
                        "ticker": "MSFT",
                        "open": "10",
                        "high": "20",
                        "low": "5",
                        "close": str(index + 10),
                        "volume": "200",
                    }
                )
            with patch.dict(
                os.environ,
                {"HOME": tmp_dir, "AUTOQUANT_WORKSPACE": str(home_path / ".nanobot" / "workspace" / "autoquant")},
                clear=False,
            ):
                with patch(
                    "autoquant_cli.quant.data.fetch_prices",
                    side_effect=lambda provider, ticker, from_date, to_date, ccxt_exchange=None: target_rows if ticker == "AAPL" else input_rows,
                ):
                    result = ensure_run_prices(
                        "run-1",
                        ["MSFT"],
                        "AAPL",
                        "2026-01-01",
                        "2026-01-05",
                        force_refresh=True,
                    )
                frame = load_dataset("run-1", min_rows=2)
                merged_rows = read_csv(prices_path("run-1"))
                raw_rows = read_csv(raw_prices_path("run-1"))
            self.assertEqual(result, "downloaded")
            self.assertEqual(len(frame), 5)
            self.assertEqual(len(merged_rows), 5)
            self.assertEqual(len(raw_rows), 10)
            self.assertIn("aapl_open", merged_rows[0])
            self.assertIn("msft_open", merged_rows[0])
            self.assertNotIn("open", merged_rows[0])
            self.assertNotIn("ticker", merged_rows[0])

    def test_ensure_run_prices_rejects_large_merge_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            home_path = Path(tmp_dir)
            started_at = datetime(2026, 1, 1, tzinfo=UTC)
            target_rows = []
            input_rows = []
            for index in range(5):
                target_ts = started_at + timedelta(hours=index)
                input_ts = target_ts + timedelta(minutes=30)
                target_rows.append(
                    {
                        "timestamp": target_ts.isoformat(),
                        "ticker": "AAPL",
                        "open": "1",
                        "high": "2",
                        "low": "0",
                        "close": str(index),
                        "volume": "100",
                    }
                )
                input_rows.append(
                    {
                        "timestamp": input_ts.isoformat(),
                        "ticker": "MSFT",
                        "open": "10",
                        "high": "20",
                        "low": "5",
                        "close": str(index + 10),
                        "volume": "200",
                    }
                )
            with patch.dict(
                os.environ,
                {"HOME": tmp_dir, "AUTOQUANT_WORKSPACE": str(home_path / ".nanobot" / "workspace" / "autoquant")},
                clear=False,
            ):
                with self.assertRaises(RuntimeError) as context:
                    with patch(
                        "autoquant_cli.quant.data.fetch_prices",
                        side_effect=lambda provider, ticker, from_date, to_date, ccxt_exchange=None: target_rows if ticker == "AAPL" else input_rows,
                    ):
                        ensure_run_prices(
                            "run-2",
                            ["MSFT"],
                            "AAPL",
                            "2026-01-01",
                            "2026-01-05",
                            force_refresh=True,
                        )
            self.assertIn("merge_asof produced no rows", str(context.exception))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import csv
import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd
from polygon import RESTClient
from sklearn.model_selection import train_test_split
from smartpy.utility.log_util import getLogger

from autoquant_cli.config import get_massive_api_key, get_workspace_root

logger = getLogger(__name__)

PRICES_RELATIVE_PATH = Path("data/prices.csv")
OHLCV_COLUMNS = ["timestamp", "ticker", "open", "high", "low", "close", "volume"]
OHLCV_REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
PRICE_FETCH_LOOKBACK_DAYS = 30
DEFAULT_TRAINING_SIZE_DAYS = 15
DEFAULT_TEST_SIZE_DAYS = 7
DEFAULT_TRAIN_TIME_LIMIT_MINUTES = 5.0
SANDBOX_RUN_ID = "sandbox"
SANDBOX_TICKER = "AAPL"
SANDBOX_FROM_DATE = "2026-02-01"
SANDBOX_TO_DATE = "2026-02-28"
CSV_LOCK = Lock()


def workspace_root() -> Path:
    path = get_workspace_root()
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_dir(run_id: str) -> Path:
    path = workspace_root() / "runs" / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def prices_path(run_id: str) -> Path:
    return run_dir(run_id) / PRICES_RELATIVE_PATH


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            return []
        return list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def upsert_csv(path: Path, fieldnames: list[str], key_fields: list[str], rows: list[dict[str, str]]) -> None:
    with CSV_LOCK:
        existing = read_csv(path)
        index: dict[tuple[str, ...], dict[str, str]] = {}
        for row in existing:
            key = tuple(row.get(name, "") for name in key_fields)
            if all(key):
                index[key] = {name: str(row.get(name, "") or "") for name in fieldnames}
        for row in rows:
            normalized = {name: str(row.get(name, "") or "") for name in fieldnames}
            key = tuple(normalized.get(name, "") for name in key_fields)
            if all(key):
                index[key] = normalized
        write_csv(path, fieldnames, list(index.values()))


def get_fetch_from_date(from_date: str) -> str:
    return (date.fromisoformat(from_date) - timedelta(days=PRICE_FETCH_LOOKBACK_DAYS)).isoformat()


def _value(item: Any, keys: list[str]) -> Any:
    for key in keys:
        if isinstance(item, dict) and key in item:
            return item[key]
        if hasattr(item, key):
            return getattr(item, key)
    return None


def _iso_utc(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp > 10_000_000_000:
            timestamp /= 1000.0
        return datetime.fromtimestamp(timestamp, tz=UTC).isoformat()
    text = str(value).replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat()


def fetch_prices(ticker: str, from_date: str, to_date: str) -> list[dict[str, str]]:
    client = RESTClient(get_massive_api_key())
    rows: list[dict[str, str]] = []
    for item in client.list_aggs(
        ticker=ticker,
        multiplier=1,
        timespan="hour",
        from_=from_date,
        to=to_date,
        limit=50000,
    ):
        timestamp = _value(item, ["timestamp", "t"])
        open_price = _value(item, ["open", "o"])
        high_price = _value(item, ["high", "h"])
        low_price = _value(item, ["low", "l"])
        close_price = _value(item, ["close", "c"])
        if timestamp is None or open_price is None or high_price is None or low_price is None or close_price is None:
            continue
        volume = _value(item, ["volume", "v"])
        rows.append(
            {
                "timestamp": _iso_utc(timestamp),
                "ticker": ticker,
                "open": str(open_price),
                "high": str(high_price),
                "low": str(low_price),
                "close": str(close_price),
                "volume": str(volume) if volume is not None else "",
            }
        )
    return rows


def ensure_run_prices(
    run_id: str,
    ticker: str,
    from_date: str,
    to_date: str,
    force_refresh: bool = False,
) -> str:
    path = prices_path(run_id)
    existing = read_csv(path)
    if existing and not force_refresh:
        return "reused"
    fetch_from = get_fetch_from_date(from_date)
    logger.info("Fetching prices for run=%s ticker=%s range=%s..%s", run_id, ticker, fetch_from, to_date)
    rows = fetch_prices(ticker, fetch_from, to_date)
    if not rows:
        raise RuntimeError(f"No OHLCV rows returned for {ticker}")
    upsert_csv(path, OHLCV_COLUMNS, ["timestamp", "ticker"], rows)
    return "downloaded"


def load_dataset(run_id: str) -> pd.DataFrame:
    frame = pd.DataFrame(read_csv(prices_path(run_id)))
    if frame.empty:
        raise RuntimeError(f"No OHLCV rows found for run_id={run_id}")
    missing_columns = [name for name in OHLCV_REQUIRED_COLUMNS if name not in frame.columns]
    if missing_columns:
        raise RuntimeError(f"OHLCV missing columns: {','.join(missing_columns)}")
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    for name in ["open", "high", "low", "close", "volume"]:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)
    if len(frame) < 220:
        raise RuntimeError("Need at least 220 OHLCV rows")
    return frame


def get_splits(frame: pd.DataFrame, feature_names: list[str], task: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    if len(frame) < 80:
        raise RuntimeError("Not enough rows to split")
    train_frame, validation_frame = train_test_split(frame, test_size=0.2, shuffle=False)
    train_frame = train_frame.reset_index(drop=True)
    validation_frame = validation_frame.reset_index(drop=True)
    if train_frame.empty or validation_frame.empty:
        raise RuntimeError("Invalid train/validation split")
    if task == "classification":
        if train_frame["target"].nunique() < 2:
            raise RuntimeError("Train target needs both classes")
        if validation_frame["target"].nunique() < 2:
            raise RuntimeError("Validation target needs both classes")
    return (
        train_frame[feature_names],
        train_frame["target"],
        validation_frame[feature_names],
        validation_frame["target"],
    )

from __future__ import annotations

import csv
import json
import logging
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd
from polygon import RESTClient
from sklearn.model_selection import train_test_split

from autoquant_cli.config import get_massive_api_key, get_workspace_root
from autoquant_cli.quant.run_metadata_validation import DEFAULT_DATA_PROVIDER, RunMarketConfig, validate_run_market_config

logger = logging.getLogger(__name__)

PRICES_RELATIVE_PATH = Path("data/prices.csv")
RAW_PRICES_RELATIVE_PATH = Path("data/raw_prices.csv")
RUN_METADATA_RELATIVE_PATH = Path("data/run_metadata.json")
OHLCV_COLUMNS = ["timestamp", "ticker", "open", "high", "low", "close", "volume"]
OHLCV_REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
DEFAULT_TRAINING_SIZE_DAYS = 15
DEFAULT_TEST_SIZE_DAYS = 7
DEFAULT_TRAIN_TIME_LIMIT_MINUTES = 5.0
HYPERPARAM_SEARCH_CANDIDATE_COUNT = 1000
HYPERPARAM_TRAINING_SIZE_DAYS_MIN = 7
HYPERPARAM_TRAINING_SIZE_DAYS_MAX = 60
MERGE_ASOF_TOLERANCE_MINUTES = 10
MAX_MERGE_DATA_LOSS_RATIO = 0.1
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


def raw_prices_path(run_id: str) -> Path:
    return run_dir(run_id) / RAW_PRICES_RELATIVE_PATH


def run_metadata_path(run_id: str) -> Path:
    return run_dir(run_id) / RUN_METADATA_RELATIVE_PATH


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


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


def read_json(path: Path) -> dict[str, object]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def save_run_market_config(run_id: str, market_config: RunMarketConfig) -> None:
    write_json(
        run_metadata_path(run_id),
        {
            "input_ohlc_tickers": market_config.input_ohlc_tickers,
            "target_ticker": market_config.target_ticker,
            "data_provider": market_config.data_provider,
            "ccxt_exchange": market_config.ccxt_exchange,
        },
    )


def load_run_market_config(run_id: str) -> RunMarketConfig:
    payload = read_json(run_metadata_path(run_id))
    if not payload:
        raise RuntimeError(f"Run market config not found for run_id={run_id}")
    return validate_run_market_config(
        input_ohlc_tickers=list(payload.get("input_ohlc_tickers") or []),
        target_ticker=str(payload.get("target_ticker") or ""),
        data_provider=str(payload.get("data_provider") or DEFAULT_DATA_PROVIDER),
        ccxt_exchange=str(payload["ccxt_exchange"]) if payload.get("ccxt_exchange") else None,
    )


def _stringify_cell(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def write_frame_csv(path: Path, frame: pd.DataFrame) -> None:
    serializable = frame.copy()
    if "timestamp" in serializable.columns:
        serializable["timestamp"] = pd.to_datetime(serializable["timestamp"], utc=True, errors="coerce")
    rows = [{column: _stringify_cell(row[column]) for column in serializable.columns} for _, row in serializable.iterrows()]
    write_csv(path, list(serializable.columns), rows)


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


def _normalize_ohlcv_row(
    ticker: str,
    timestamp: Any,
    open_price: Any,
    high_price: Any,
    low_price: Any,
    close_price: Any,
    volume: Any,
) -> dict[str, str]:
    return {
        "timestamp": _iso_utc(timestamp),
        "ticker": ticker,
        "open": str(open_price),
        "high": str(high_price),
        "low": str(low_price),
        "close": str(close_price),
        "volume": str(volume) if volume is not None else "",
    }


def fetch_prices_from_massive(ticker: str, from_date: str, to_date: str) -> list[dict[str, str]]:
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
        rows.append(_normalize_ohlcv_row(ticker, timestamp, open_price, high_price, low_price, close_price, volume))
    return rows


def fetch_prices_from_ccxt(ticker: str, from_date: str, to_date: str, ccxt_exchange: str | None) -> list[dict[str, str]]:
    try:
        import ccxt
    except ImportError as exc:
        raise RuntimeError("ccxt is not installed") from exc
    exchange_name = str(ccxt_exchange or "binance").strip().lower()
    exchange_class = getattr(ccxt, exchange_name, None)
    if exchange_class is None:
        raise RuntimeError(f"Unknown ccxt exchange: {exchange_name}")
    exchange = exchange_class({"enableRateLimit": True})
    since_ms = int(datetime.fromisoformat(f"{from_date}T00:00:00+00:00").timestamp() * 1000)
    end_ms = int((datetime.fromisoformat(f"{to_date}T00:00:00+00:00") + timedelta(days=1)).timestamp() * 1000)
    rows: list[dict[str, str]] = []
    while since_ms < end_ms:
        batch = exchange.fetch_ohlcv(ticker, timeframe="1h", since=since_ms, limit=1000)
        if not batch:
            break
        last_timestamp = since_ms
        for candle in batch:
            timestamp_ms, open_price, high_price, low_price, close_price, volume = candle[:6]
            if int(timestamp_ms) >= end_ms:
                continue
            rows.append(_normalize_ohlcv_row(ticker, int(timestamp_ms), open_price, high_price, low_price, close_price, volume))
            last_timestamp = max(last_timestamp, int(timestamp_ms))
        if last_timestamp <= since_ms:
            break
        since_ms = last_timestamp + 1
    return rows


def fetch_prices(
    data_provider: str,
    ticker: str,
    from_date: str,
    to_date: str,
    ccxt_exchange: str | None = None,
) -> list[dict[str, str]]:
    if data_provider == "massive":
        return fetch_prices_from_massive(ticker, from_date, to_date)
    if data_provider == "ccxt":
        return fetch_prices_from_ccxt(ticker, from_date, to_date, ccxt_exchange)
    raise RuntimeError(f"Unsupported data provider: {data_provider}")


def ensure_run_prices(
    run_id: str,
    input_ohlc_tickers: list[str],
    target_ticker: str,
    from_date: str,
    to_date: str,
    data_provider: str = DEFAULT_DATA_PROVIDER,
    ccxt_exchange: str | None = None,
    force_refresh: bool = False,
) -> str:
    market_config = validate_run_market_config(
        input_ohlc_tickers=input_ohlc_tickers,
        target_ticker=target_ticker,
        data_provider=data_provider,
        ccxt_exchange=ccxt_exchange,
    )
    path = raw_prices_path(run_id)
    metadata_path = run_metadata_path(run_id)
    existing = read_csv(path)
    if existing and metadata_path.exists():
        existing_config = load_run_market_config(run_id)
        if existing_config != market_config:
            force_refresh = True
    save_run_market_config(run_id, market_config)
    existing_tickers = {str(row.get("ticker") or "") for row in existing}
    requested_tickers = market_config.all_tickers
    tickers_to_fetch = requested_tickers if force_refresh else [ticker for ticker in requested_tickers if ticker not in existing_tickers]
    if existing and not tickers_to_fetch:
        merged = _build_merged_dataset(_clean_long_ohlcv_frame(run_id), market_config)
        write_frame_csv(prices_path(run_id), merged)
        return "reused"
    all_rows: list[dict[str, str]] = []
    for ticker in tickers_to_fetch:
        logger.info("Fetching prices for run=%s ticker=%s range=%s..%s provider=%s", run_id, ticker, from_date, to_date, market_config.data_provider)
        rows = fetch_prices(
            market_config.data_provider,
            ticker,
            from_date,
            to_date,
            ccxt_exchange=market_config.ccxt_exchange,
        )
        if not rows:
            raise RuntimeError(f"No OHLCV rows returned for {ticker}")
        all_rows.extend(rows)
    if force_refresh:
        write_csv(path, OHLCV_COLUMNS, all_rows)
    else:
        upsert_csv(path, OHLCV_COLUMNS, ["timestamp", "ticker"], all_rows)
    merged = _build_merged_dataset(_clean_long_ohlcv_frame(run_id), market_config)
    write_frame_csv(prices_path(run_id), merged)
    return "downloaded"


def _sanitize_ticker_prefix(ticker: str) -> str:
    prefix = re.sub(r"[^0-9A-Za-z]+", "_", ticker).strip("_").lower()
    return prefix or "ticker"


def _prefixed_ohlcv_columns(ticker: str) -> dict[str, str]:
    prefix = _sanitize_ticker_prefix(ticker)
    return {name: f"{prefix}_{name}" for name in ["open", "high", "low", "close", "volume"]}


def _clean_long_ohlcv_frame(run_id: str) -> pd.DataFrame:
    frame = pd.DataFrame(read_csv(raw_prices_path(run_id)))
    if frame.empty:
        raise RuntimeError(f"No OHLCV rows found for run_id={run_id}")
    missing_columns = [name for name in OHLCV_REQUIRED_COLUMNS if name not in frame.columns]
    if missing_columns:
        raise RuntimeError(f"OHLCV missing columns: {','.join(missing_columns)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).reset_index(drop=True)
    for name in ["open", "high", "low", "close", "volume"]:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)
    frame = frame.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    return frame


def _clean_merged_frame(run_id: str) -> pd.DataFrame:
    frame = pd.DataFrame(read_csv(prices_path(run_id)))
    if frame.empty:
        raise RuntimeError(f"No merged OHLCV rows found for run_id={run_id}")
    market_config = load_run_market_config(run_id)
    target_columns = _prefixed_ohlcv_columns(market_config.target_ticker)
    missing_columns = [name for name in ["timestamp", *target_columns.values()] if name not in frame.columns]
    if missing_columns:
        raise RuntimeError(f"OHLCV missing columns: {','.join(missing_columns)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).reset_index(drop=True)
    numeric_columns = [column for column in frame.columns if column != "timestamp"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=numeric_columns).sort_values("timestamp").reset_index(drop=True)
    for name, column in target_columns.items():
        frame[name] = frame[column]
    return frame


def _build_target_frame(frame: pd.DataFrame, target_ticker: str) -> pd.DataFrame:
    target_frame = frame[frame["ticker"] == target_ticker][["timestamp", "open", "high", "low", "close", "volume"]].copy()
    if target_frame.empty:
        raise RuntimeError(f"No OHLCV rows found for target_ticker={target_ticker}")
    target_frame = target_frame.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)
    return target_frame.rename(columns=_prefixed_ohlcv_columns(target_ticker))


def _merge_input_ticker(base_frame: pd.DataFrame, frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    input_frame = frame[frame["ticker"] == ticker][["timestamp", "open", "high", "low", "close", "volume"]].copy()
    if input_frame.empty:
        raise RuntimeError(f"No OHLCV rows found for input ticker {ticker}")
    input_frame = input_frame.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)
    prefix = _sanitize_ticker_prefix(ticker)
    renamed = input_frame.rename(
        columns={
            "timestamp": f"{prefix}_timestamp",
            **_prefixed_ohlcv_columns(ticker),
        }
    )
    merged = pd.merge_asof(
        base_frame.sort_values("timestamp"),
        renamed.sort_values(f"{prefix}_timestamp"),
        left_on="timestamp",
        right_on=f"{prefix}_timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=MERGE_ASOF_TOLERANCE_MINUTES),
    )
    merged = merged.dropna(
        subset=[
            f"{prefix}_timestamp",
            f"{prefix}_open",
            f"{prefix}_high",
            f"{prefix}_low",
            f"{prefix}_close",
            f"{prefix}_volume",
        ]
    ).reset_index(drop=True)
    if merged.empty:
        raise RuntimeError(f"merge_asof produced no rows for input ticker {ticker}")
    max_delta = (merged["timestamp"] - merged[f"{prefix}_timestamp"]).abs().max()
    if pd.notna(max_delta) and max_delta > pd.Timedelta(minutes=MERGE_ASOF_TOLERANCE_MINUTES):
        raise RuntimeError(f"Merged data contains gaps larger than {MERGE_ASOF_TOLERANCE_MINUTES} minutes for input ticker {ticker}")
    return merged.drop(columns=[f"{prefix}_timestamp"])


def _build_merged_dataset(frame: pd.DataFrame, market_config: RunMarketConfig) -> pd.DataFrame:
    target_frame = _build_target_frame(frame, market_config.target_ticker)
    original_target_rows = len(target_frame)
    merged = target_frame
    for ticker in market_config.input_ohlc_tickers:
        merged = _merge_input_ticker(merged, frame, ticker)
    retained_ratio = len(merged) / original_target_rows if original_target_rows else 0.0
    if original_target_rows and (1.0 - retained_ratio) > MAX_MERGE_DATA_LOSS_RATIO:
        raise RuntimeError("Merged dataset lost more than 10% of target rows")
    return merged.sort_values("timestamp").reset_index(drop=True)


def load_dataset(run_id: str, min_rows: int = 220) -> pd.DataFrame:
    merged = _clean_merged_frame(run_id)
    if len(merged) < min_rows:
        raise RuntimeError(f"Need at least {min_rows} OHLCV rows")
    return merged


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

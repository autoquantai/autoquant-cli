from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from autoquant_cli.commands.api_client import get_run
from autoquant_cli.quant.data import (
    DEFAULT_TEST_SIZE_DAYS,
    DEFAULT_TRAINING_SIZE_DAYS,
    ensure_run_prices,
    prices_path,
    read_csv,
    run_dir,
)
from autoquant_cli.quant.runtime import run_train_file

logger = logging.getLogger(__name__)


def _selected_training_size_days(output: dict[str, object]) -> int:
    train = output.get("train")
    if not isinstance(train, dict):
        return DEFAULT_TRAINING_SIZE_DAYS
    selected_hyperparams = train.get("selected_hyperparams")
    if not isinstance(selected_hyperparams, dict):
        return DEFAULT_TRAINING_SIZE_DAYS
    value = selected_hyperparams.get("training_size_days")
    if value is None:
        return DEFAULT_TRAINING_SIZE_DAYS
    return int(value)


def _build_error(output: dict[str, object]) -> str | None:
    runtime_error = str(output.get("runtime_error") or "").strip()
    stderr = str(output.get("stderr") or "").strip()
    stdout = str(output.get("stdout") or "").strip()
    if not runtime_error and not stderr:
        return None
    parts = [part for part in [runtime_error, stderr, stdout] if part]
    return "\n\n".join(parts)


def validate_model(
    model_path: str,
    run_id: str,
    task: str | None = None,
    refresh_data: bool = False,
) -> dict[str, Any]:
    source_path = Path(model_path)
    if not source_path.exists():
        raise RuntimeError(f"Model file not found: {source_path}")
    run = get_run(run_id)
    resolved_task = task or str(run["task"])
    sandbox_rows = read_csv(prices_path(run_id))
    initialized = not bool(sandbox_rows)
    input_ohlc_tickers = list(run.get("input_ohlc_tickers") or [])
    target_ticker = str(run["target_ticker"])
    data_provider = str(run.get("data_provider") or "massive")
    ccxt_exchange = str(run["ccxt_exchange"]) if run.get("ccxt_exchange") else None
    data_source = ensure_run_prices(
        run_id,
        input_ohlc_tickers,
        target_ticker,
        str(run["from_date"]),
        str(run["to_date"]),
        data_provider=data_provider,
        ccxt_exchange=ccxt_exchange,
        force_refresh=refresh_data,
    )
    output = run_train_file(
        source_path,
        run_id=run_id,
        model_id=source_path.stem,
        expected_task=resolved_task,
        training_size_days=DEFAULT_TRAINING_SIZE_DAYS,
        execution_profile="sandbox",
    )
    selected_training_size_days = _selected_training_size_days(output)
    error = _build_error(output)
    validation_metrics = output.get("validation")
    status = "completed"
    metrics = validation_metrics if isinstance(validation_metrics, dict) else None
    if error or not isinstance(validation_metrics, dict):
        status = "failed"
        if error is None:
            error = f"Invalid train output: {output}"
    logger.info("Validation finished model=%s status=%s", source_path.name, status)
    return {
        "validation_run_id": run_id,
        "validation_run_dir": str(run_dir(run_id)),
        "model_id": source_path.stem,
        "status": status,
        "metrics": metrics,
        "error": error,
        "task": resolved_task,
        "input_ohlc_tickers": input_ohlc_tickers,
        "target_ticker": target_ticker,
        "data_provider": data_provider,
        "ccxt_exchange": ccxt_exchange,
        "from_date": str(run["from_date"]),
        "to_date": str(run["to_date"]),
        "training_size_days": selected_training_size_days,
        "test_size_days": DEFAULT_TEST_SIZE_DAYS,
        "data_source": data_source,
        "sandbox_initialized": initialized,
    }

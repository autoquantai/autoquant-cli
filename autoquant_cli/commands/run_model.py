from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from autoquant_cli.commands.api_client import get_run, post_json
from autoquant_cli.quant.data import DEFAULT_TRAINING_SIZE_DAYS, ensure_run_prices
from autoquant_cli.quant.run_metadata_validation import validate_run_market_config
from autoquant_cli.quant.runtime import run_train_file

logger = logging.getLogger(__name__)


def _selected_hyperparameters(output: dict[str, object]) -> dict[str, object]:
    train = output.get("train")
    if not isinstance(train, dict):
        return {}
    selected_hyperparams = train.get("selected_hyperparams")
    if not isinstance(selected_hyperparams, dict):
        return {}
    return dict(selected_hyperparams)


def _build_error(output: dict[str, object]) -> str | None:
    runtime_error = str(output.get("runtime_error") or "").strip()
    stderr = str(output.get("stderr") or "").strip()
    stdout = str(output.get("stdout") or "").strip()
    if not runtime_error and not stderr:
        return None
    parts = [part for part in [runtime_error, stderr, stdout] if part]
    return "\n\n".join(parts)


def _build_evals(output: dict[str, object]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key in ("train", "validation", "stdout", "stderr", "runtime_error"):
        value = output.get(key)
        if value not in (None, ""):
            payload[key] = value
    return payload


def run_model(
    run_id: str,
    name: str,
    generation: int,
    model_path: str,
    log: str,
    parent_ids: list[str] | None = None,
    reasoning: str = "",
    task: str | None = None,
) -> dict[str, Any]:
    resolved_parent_ids = parent_ids or ["seed"]
    if len(resolved_parent_ids) < 1 or len(resolved_parent_ids) > 2:
        raise RuntimeError("parent_ids must contain between 1 and 2 items")
    source_path = Path(model_path)
    if not source_path.exists():
        raise RuntimeError(f"Model file not found: {source_path}")
    run = get_run(run_id)
    resolved_task = task or str(run["task"])
    market_config = validate_run_market_config(
        input_ohlc_tickers=list(run.get("input_ohlc_tickers") or []),
        target_ticker=str(run["target_ticker"]),
        data_provider=str(run.get("data_provider") or "massive"),
        ccxt_exchange=str(run["ccxt_exchange"]) if run.get("ccxt_exchange") else None,
    )
    data_source = ensure_run_prices(
        run_id,
        market_config.input_ohlc_tickers,
        market_config.target_ticker,
        str(run["from_date"]),
        str(run["to_date"]),
        data_provider=market_config.data_provider,
        ccxt_exchange=market_config.ccxt_exchange,
    )
    output = run_train_file(
        source_path,
        run_id=run_id,
        model_id=source_path.stem,
        expected_task=resolved_task,
        training_size_days=DEFAULT_TRAINING_SIZE_DAYS,
        train_time_limit_minutes=float(run.get("train_time_limit_minutes") or 5.0),
    )
    selected_hyperparameters = _selected_hyperparameters(output)
    source = source_path.read_text(encoding="utf-8")
    payload = {
        "run_id": run_id,
        "name": name,
        "generation": generation,
        "model": {
            "name": name,
            "filepath": str(source_path),
            "source": source,
            "task": resolved_task,
            "hyperparameters": selected_hyperparameters,
        },
        "parent_ids": resolved_parent_ids,
        "log": log,
        "reasoning": reasoning,
        "evals": _build_evals(output),
        "error": _build_error(output),
    }
    response = post_json("/experiment/create", payload)
    if isinstance(response, dict):
        response["data_source"] = data_source
    logger.info("Created experiment run_id=%s generation=%s", run_id, generation)
    return response

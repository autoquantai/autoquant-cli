from __future__ import annotations

import logging
from typing import Any

from autoquant_cli.commands.api_client import post_json
from autoquant_cli.quant.data import ensure_run_prices, run_dir
from autoquant_cli.quant.experiment_date_validation import validate_experiment_dates
from autoquant_cli.quant.run_metadata_validation import validate_run_market_config

logger = logging.getLogger(__name__)


def create_experiment(
    name: str,
    input_ohlc_tickers: list[str],
    target_ticker: str,
    from_date: str,
    to_date: str,
    task: str,
    max_experiments: int,
    data_provider: str = "massive",
    ccxt_exchange: str | None = None,
    train_time_limit_minutes: int = 30,
    refresh_data: bool = False,
) -> dict[str, Any]:
    validate_experiment_dates(from_date, to_date)
    market_config = validate_run_market_config(
        input_ohlc_tickers=input_ohlc_tickers,
        target_ticker=target_ticker,
        data_provider=data_provider,
        ccxt_exchange=ccxt_exchange,
    )
    payload = {
        "name": name,
        "input_ohlc_tickers": market_config.input_ohlc_tickers,
        "target_ticker": market_config.target_ticker,
        "data_provider": market_config.data_provider,
        "ccxt_exchange": market_config.ccxt_exchange,
        "from_date": from_date,
        "to_date": to_date,
        "task": task,
        "max_experiments": max_experiments,
        "train_time_limit_minutes": train_time_limit_minutes,
    }
    response = post_json("/run/create", payload)
    if not isinstance(response, dict):
        raise RuntimeError(f"Unexpected run create response: {response}")
    run_id = str(response.get("id") or "").strip()
    if not run_id:
        raise RuntimeError(f"Run create response missing id: {response}")
    local_run_dir = run_dir(run_id)
    data_source = ensure_run_prices(
        run_id,
        market_config.input_ohlc_tickers,
        market_config.target_ticker,
        from_date,
        to_date,
        data_provider=market_config.data_provider,
        ccxt_exchange=market_config.ccxt_exchange,
        force_refresh=refresh_data,
    )
    result = dict(response)
    result["run_dir"] = str(local_run_dir)
    result["data_source"] = data_source
    logger.info("Created experiment id=%s target_ticker=%s", run_id, market_config.target_ticker)
    return result

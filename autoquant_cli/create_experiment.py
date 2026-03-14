from __future__ import annotations

import logging
from typing import Any

from autoquant_cli.api_client import post_json
from autoquant_cli.data import ensure_run_prices, get_fetch_from_date, run_dir

logger = logging.getLogger(__name__)


def create_experiment(
    name: str,
    ticker: str,
    from_date: str,
    to_date: str,
    task: str,
    max_experiments: int,
    train_time_limit_minutes: int = 30,
    refresh_data: bool = False,
) -> dict[str, Any]:
    payload = {
        "name": name,
        "ticker": ticker,
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
        ticker,
        from_date,
        to_date,
        force_refresh=refresh_data,
    )
    result = dict(response)
    result["run_dir"] = str(local_run_dir)
    result["fetch_from_date"] = get_fetch_from_date(from_date)
    result["data_source"] = data_source
    logger.info("Created experiment id=%s ticker=%s", run_id, ticker)
    return result

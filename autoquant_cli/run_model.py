from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from autoquant_cli.api_client import get_run, post_json
from autoquant_cli.data import DEFAULT_TEST_SIZE_DAYS, DEFAULT_TRAINING_SIZE_DAYS, ensure_run_prices
from autoquant_cli.runtime import run_train_file

logger = logging.getLogger(__name__)


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
    parent_id: str | None = None,
    reasoning: str = "",
    task: str | None = None,
) -> dict[str, Any]:
    source_path = Path(model_path)
    if not source_path.exists():
        raise RuntimeError(f"Model file not found: {source_path}")
    run = get_run(run_id)
    resolved_task = task or str(run["task"])
    data_source = ensure_run_prices(
        run_id,
        str(run["ticker"]),
        str(run["from_date"]),
        str(run["to_date"]),
    )
    output = run_train_file(
        source_path,
        run_id=run_id,
        model_id=source_path.stem,
        expected_task=resolved_task,
        training_size_days=DEFAULT_TRAINING_SIZE_DAYS,
        test_size_days=DEFAULT_TEST_SIZE_DAYS,
        train_time_limit_minutes=float(run.get("train_time_limit_minutes") or 5.0),
    )
    source = source_path.read_text(encoding="utf-8")
    payload = {
        "run_id": run_id,
        "name": name,
        "generation": generation,
        "model": {
            "name": name,
            "filename": source_path.name,
            "source": source,
            "task": resolved_task,
            "training_size_days": DEFAULT_TRAINING_SIZE_DAYS,
            "test_size_days": DEFAULT_TEST_SIZE_DAYS,
        },
        "parent_id": parent_id,
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

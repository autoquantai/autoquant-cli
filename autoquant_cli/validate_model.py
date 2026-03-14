from __future__ import annotations

from pathlib import Path
from typing import Any

from smartpy.utility.log_util import getLogger

from autoquant_cli.data import (
    DEFAULT_TEST_SIZE_DAYS,
    DEFAULT_TRAINING_SIZE_DAYS,
    SANDBOX_FROM_DATE,
    SANDBOX_RUN_ID,
    SANDBOX_TICKER,
    SANDBOX_TO_DATE,
    ensure_run_prices,
    prices_path,
    read_csv,
    run_dir,
)
from autoquant_cli.runtime import run_train_file

logger = getLogger(__name__)


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
    task: str,
    refresh_data: bool = False,
) -> dict[str, Any]:
    source_path = Path(model_path)
    if not source_path.exists():
        raise RuntimeError(f"Model file not found: {source_path}")
    sandbox_rows = read_csv(prices_path(SANDBOX_RUN_ID))
    initialized = not bool(sandbox_rows)
    data_source = ensure_run_prices(
        SANDBOX_RUN_ID,
        SANDBOX_TICKER,
        SANDBOX_FROM_DATE,
        SANDBOX_TO_DATE,
        force_refresh=refresh_data,
    )
    output = run_train_file(
        source_path,
        run_id=SANDBOX_RUN_ID,
        model_id=source_path.stem,
        expected_task=task,
        training_size_days=DEFAULT_TRAINING_SIZE_DAYS,
        test_size_days=DEFAULT_TEST_SIZE_DAYS,
    )
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
        "validation_run_id": SANDBOX_RUN_ID,
        "validation_run_dir": str(run_dir(SANDBOX_RUN_ID)),
        "model_id": source_path.stem,
        "status": status,
        "metrics": metrics,
        "error": error,
        "task": task,
        "ticker": SANDBOX_TICKER,
        "from_date": SANDBOX_FROM_DATE,
        "to_date": SANDBOX_TO_DATE,
        "training_size_days": DEFAULT_TRAINING_SIZE_DAYS,
        "test_size_days": DEFAULT_TEST_SIZE_DAYS,
        "data_source": data_source,
        "sandbox_initialized": initialized,
    }

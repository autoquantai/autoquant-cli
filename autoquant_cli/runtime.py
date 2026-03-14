from __future__ import annotations

import io
import traceback
from contextlib import redirect_stderr, redirect_stdout
from inspect import isabstract
from pathlib import Path

from smartpy.utility.log_util import getLogger

from autoquant_cli.data import DEFAULT_TEST_SIZE_DAYS, DEFAULT_TRAINING_SIZE_DAYS, DEFAULT_TRAIN_TIME_LIMIT_MINUTES
from autoquant_cli.model_base import AutoQuantModel

logger = getLogger(__name__)


def _validate_metrics_payload(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise RuntimeError("Model run output must be a dict")
    keys = set(payload.keys())
    if keys != {"train", "validation"}:
        raise RuntimeError("Model run output must contain exactly train and validation keys")
    if not isinstance(payload.get("train"), dict) or not isinstance(payload.get("validation"), dict):
        raise RuntimeError("Model run output train and validation values must be dicts")
    return payload


def _discover_model_class(env: dict[str, object], module_name: str) -> type[AutoQuantModel]:
    classes = [
        value
        for value in env.values()
        if isinstance(value, type)
        and issubclass(value, AutoQuantModel)
        and value is not AutoQuantModel
        and value.__module__ == module_name
    ]
    if not classes:
        raise RuntimeError("Model file must define exactly one concrete AutoQuantModel subclass, found 0")
    concrete = [cls for cls in classes if not isabstract(cls)]
    if len(concrete) == 1:
        return concrete[0]
    if len(concrete) > 1:
        names = ", ".join(sorted(cls.__name__ for cls in concrete))
        raise RuntimeError(f"Model file must define exactly one concrete AutoQuantModel subclass, found {len(concrete)}: {names}")
    details: list[str] = []
    for cls in classes:
        missing = sorted(getattr(cls, "__abstractmethods__", set()))
        if missing:
            details.append(f"{cls.__name__} missing {', '.join(missing)}")
        else:
            details.append(cls.__name__)
    raise RuntimeError(f"Model file has no concrete AutoQuantModel subclass. Abstract subclasses found: {'; '.join(details)}")


def run_train_file(
    path: Path,
    run_id: str,
    model_id: str | None = None,
    expected_task: str | None = None,
    training_size_days: int | None = None,
    test_size_days: int | None = None,
    train_time_limit_minutes: float | None = None,
) -> dict[str, object]:
    source = path.read_text(encoding="utf-8")
    code = compile(source, str(path), "exec")
    module_name = "__autoquant_model__"
    env: dict[str, object] = {"__name__": module_name, "__file__": str(path)}
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    output: dict[str, object] = {}
    runtime_error = ""
    logger.info("Executing model file %s", path)
    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            exec(code, env, env)
            model_class = _discover_model_class(env, module_name)
            model = model_class(
                run_id=run_id,
                task=expected_task or "classification",
                model_id=model_id or path.stem,
                model_path=str(path),
            )
            payload = model.run(
                training_size_days=training_size_days or DEFAULT_TRAINING_SIZE_DAYS,
                test_size_days=test_size_days or DEFAULT_TEST_SIZE_DAYS,
                train_time_limit_minutes=train_time_limit_minutes or DEFAULT_TRAIN_TIME_LIMIT_MINUTES,
            )
            validated = _validate_metrics_payload(payload)
            output["train"] = validated["train"]
            output["validation"] = validated["validation"]
    except BaseException:
        runtime_error = traceback.format_exc()
    output["stdout"] = stdout_buffer.getvalue()
    output["stderr"] = stderr_buffer.getvalue()
    if runtime_error:
        output["runtime_error"] = runtime_error
    return output

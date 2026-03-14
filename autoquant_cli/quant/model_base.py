from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from numbers import Integral, Real
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.metrics import (
    classification_report,
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import ParameterSampler

from autoquant_cli.quant.data import (
    DEFAULT_TEST_SIZE_DAYS,
    HYPERPARAM_SEARCH_CANDIDATE_COUNT,
    HYPERPARAM_TRAINING_SIZE_DAYS_MAX,
    HYPERPARAM_TRAINING_SIZE_DAYS_MIN,
    get_splits,
    load_dataset,
)

logger = logging.getLogger(__name__)
SANDBOX_SAMPLE_ROWS = 72


def walk_forward(
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    training_size_days: int,
    test_size_days: int = DEFAULT_TEST_SIZE_DAYS,
):
    test_start_ts = start_ts + pd.Timedelta(days=training_size_days)
    while test_start_ts < end_ts:
        test_end_ts = min(test_start_ts + pd.Timedelta(days=test_size_days), end_ts + pd.Timedelta(microseconds=1))
        train_start_ts = test_start_ts - pd.Timedelta(days=training_size_days)
        yield train_start_ts, test_start_ts, test_end_ts
        test_start_ts = test_end_ts


def _regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    true_values = [float(value) for value in y_true]
    pred_values = [float(value) for value in y_pred]
    if len(true_values) != len(pred_values):
        raise RuntimeError("y_true and y_pred must have equal length")
    if not true_values:
        raise RuntimeError("y_true and y_pred cannot be empty")
    mse = mean_squared_error(true_values, pred_values)
    return {
        "n_samples": float(len(true_values)),
        "mae": float(mean_absolute_error(true_values, pred_values)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(true_values, pred_values)),
        "explained_variance": float(explained_variance_score(true_values, pred_values)),
        "median_ae": float(median_absolute_error(true_values, pred_values)),
        "max_error": float(max_error(true_values, pred_values)),
    }



class AutoQuantModel(ABC):
    def __init__(
        self,
        run_id: str,
        task: str,
        model_id: str | None = None,
        model_path: str | None = None,
    ):
        self.run_id = run_id
        self.task = task
        self.model_id = model_id or self._derive_model_id(model_path)
        self.artifacts: dict[str, object] = {}
        self.best_hyperparams: dict[str, object] = {}
        self.train_metrics: dict[str, object] | None = None
        self.selected_training_size_days: int | None = None

    def _derive_model_id(self, model_path: str | None) -> str:
        if not model_path:
            return "unknown"
        stem = Path(model_path).stem.strip()
        return stem or "unknown"

    def prepare_data(self, min_rows: int = 220) -> pd.DataFrame:
        frame = load_dataset(self.run_id, min_rows=min_rows)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["timestamp"]).reset_index(drop=True)
        if frame.empty:
            raise RuntimeError("Dataset is empty")
        return frame

    @abstractmethod
    def create_features(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        raise NotImplementedError

    def split_data(self, frame: pd.DataFrame, feature_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        x_train, y_train, x_validation, y_validation = get_splits(frame, feature_names, self.task)
        split_idx = len(x_train)
        train_frame = frame.iloc[:split_idx].reset_index(drop=True)
        validation_frame = frame.iloc[split_idx:].reset_index(drop=True)
        if len(train_frame) != len(y_train) or len(validation_frame) != len(y_validation):
            raise RuntimeError("Split alignment error")
        return train_frame, validation_frame

    def evaluate(
        self,
        task: str,
        train_actual: Sequence[float | int],
        train_pred: Sequence[float | int],
        validation_actual: Sequence[float | int],
        validation_pred: Sequence[float | int],
    ) -> dict[str, object]:
        if task == "classification":
            if len(train_actual) != len(train_pred) or len(validation_actual) != len(validation_pred):
                raise RuntimeError("y_true and y_pred must have equal length")
            if not train_actual or not validation_actual:
                raise RuntimeError("y_true and y_pred cannot be empty")
            train_report = classification_report(train_actual, train_pred, output_dict=True, zero_division=0)
            validation_report = classification_report(validation_actual, validation_pred, output_dict=True, zero_division=0)
            return {"train": train_report, "validation": validation_report}
        if task == "regression":
            train_metrics = _regression_metrics([float(value) for value in train_actual], [float(value) for value in train_pred])
            validation_metrics = _regression_metrics(
                [float(value) for value in validation_actual],
                [float(value) for value in validation_pred],
            )
            return {"train": train_metrics, "validation": validation_metrics}
        raise RuntimeError("task must be classification or regression")


    def _normalize_hyperparameter_space(self, search_space: dict[str, object]) -> dict[str, object]:
        normalized: dict[str, object] = {}
        for name, spec in search_space.items():
            if isinstance(spec, range):
                values = list(spec)
                if not values:
                    raise RuntimeError("Hyperparameter range cannot be empty")
                normalized[name] = values
                continue
            if isinstance(spec, list):
                if not spec:
                    raise RuntimeError("Hyperparameter choices cannot be empty")
                normalized[name] = spec
                continue
            if isinstance(spec, tuple):
                if len(spec) == 2 and all(isinstance(value, Real) and not isinstance(value, bool) for value in spec):
                    lower, upper = spec
                    if float(lower) > float(upper):
                        raise RuntimeError("Hyperparameter numeric range must be ordered")
                    if float(lower) == float(upper):
                        normalized[name] = [lower]
                        continue
                    if all(isinstance(value, Integral) and not isinstance(value, bool) for value in spec):
                        normalized[name] = randint(int(lower), int(upper) + 1)
                        continue
                    normalized[name] = uniform(float(lower), float(upper) - float(lower))
                    continue
                if not spec:
                    raise RuntimeError("Hyperparameter choices cannot be empty")
                normalized[name] = list(spec)
                continue
            normalized[name] = [spec]
        return normalized

    def _build_hyperparameter_candidates(self) -> list[dict[str, object]]:
        search_space = self.get_hyperparameter_candidates() or {}
        if not isinstance(search_space, dict):
            raise RuntimeError("get_hyperparameter_candidates must return a dict[str, object]")
        candidate_space = dict(search_space)
        if "training_size_days" in candidate_space:
            raise RuntimeError("training_size_days is reserved and added automatically")
        candidate_space["training_size_days"] = (
            HYPERPARAM_TRAINING_SIZE_DAYS_MIN,
            HYPERPARAM_TRAINING_SIZE_DAYS_MAX,
        )
        normalized_space = self._normalize_hyperparameter_space(candidate_space)
        return list(ParameterSampler(normalized_space, n_iter=HYPERPARAM_SEARCH_CANDIDATE_COUNT, random_state=42))

    def _build_single_hyperparameter_candidate(self) -> dict[str, object]:
        search_space = self.get_hyperparameter_candidates() or {}
        if not isinstance(search_space, dict):
            raise RuntimeError("get_hyperparameter_candidates must return a dict[str, object]")
        if "training_size_days" in search_space:
            raise RuntimeError("training_size_days is reserved and added automatically")
        if not search_space:
            return {}
        normalized_space = self._normalize_hyperparameter_space(dict(search_space))
        return list(ParameterSampler(normalized_space, n_iter=1, random_state=42))[0]

    def get_hyperparameter_candidates(self) -> dict[str, object]:
        return {}

    @abstractmethod
    def fit(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        hyperparams: dict[str, object],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, x_test: pd.DataFrame) -> list[float | int]:
        raise NotImplementedError

    def _selection_score(self, metrics: dict[str, object]) -> float:
        if self.task == "classification":
            weighted_avg = metrics.get("weighted avg")
            if not isinstance(weighted_avg, dict):
                raise RuntimeError("classification metrics must include weighted avg")
            return float(weighted_avg["f1-score"])
        return float(metrics["r2"])

    def _walk_forward_predict(
        self,
        frame: pd.DataFrame,
        feature_names: list[str],
        training_size_days: int,
        hyperparams: dict[str, object],
        test_range_start_ts: pd.Timestamp | None = None,
        test_range_end_ts: pd.Timestamp | None = None,
    ) -> tuple[list[float | int], list[float | int]]:
        if frame.empty:
            raise RuntimeError("Partition is empty")
        if test_range_start_ts is not None and test_range_end_ts is not None:
            start_ts = test_range_start_ts
            end_ts = test_range_end_ts
        else:
            start_ts = frame["timestamp"].min()
            end_ts = frame["timestamp"].max()

        actual: list[float | int] = []
        predicted: list[float | int] = []
        for train_start_ts, test_start_ts, test_end_ts in walk_forward(
            start_ts=start_ts,
            end_ts=end_ts,
            training_size_days=training_size_days,
        ):
            train_window = frame[(frame["timestamp"] >= train_start_ts) & (frame["timestamp"] < test_start_ts)]
            test_window = frame[(frame["timestamp"] >= test_start_ts) & (frame["timestamp"] < test_end_ts)]
            if train_window.empty or test_window.empty:
                continue
            x_train = train_window[feature_names]
            y_train = train_window["target"]
            x_test = test_window[feature_names]
            y_test = test_window["target"]
            if self.task == "classification" and y_train.nunique() < 2:
                continue
            self.artifacts = {}
            self.fit(x_train, y_train, hyperparams)
            y_pred = list(self.predict(x_test))
            if len(y_pred) != len(x_test):
                raise RuntimeError("predict output length must match x_test length")
            if self.task == "classification":
                predicted.extend([int(value) for value in y_pred])
                actual.extend(y_test.astype(int).tolist())
            else:
                predicted.extend([float(value) for value in y_pred])
                actual.extend(y_test.astype(float).tolist())
        if not predicted:
            raise RuntimeError("Walk-forward produced no predictions")
        return actual, predicted

    def _metrics_from_predictions(
        self,
        actual: list[float | int],
        predicted: list[float | int],
    ) -> dict[str, object]:
        paired = self.evaluate(self.task, actual, predicted, actual, predicted)
        metrics = paired["train"]
        if not isinstance(metrics, dict):
            raise RuntimeError("Invalid metrics payload from evaluate")
        return metrics

    def _split_sandbox_data(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if len(frame) < 2:
            raise RuntimeError("Sandbox dataset must contain at least 2 rows")
        split_idx = max(1, len(frame) - max(1, len(frame) // 3))
        split_idx = min(split_idx, len(frame) - 1)
        train_frame = frame.iloc[:split_idx].reset_index(drop=True)
        validation_frame = frame.iloc[split_idx:].reset_index(drop=True)
        if train_frame.empty or validation_frame.empty:
            raise RuntimeError("Sandbox split produced an empty partition")
        if self.task == "classification" and train_frame["target"].nunique() < 2:
            raise RuntimeError("Sandbox train target needs both classes")
        return train_frame, validation_frame

    def _fit_predict_once(
        self,
        train_frame: pd.DataFrame,
        validation_frame: pd.DataFrame,
        feature_names: list[str],
        hyperparams: dict[str, object],
    ) -> tuple[list[float | int], list[float | int], list[float | int], list[float | int]]:
        x_train = train_frame[feature_names]
        y_train = train_frame["target"]
        x_validation = validation_frame[feature_names]
        y_validation = validation_frame["target"]
        self.artifacts = {}
        self.fit(x_train, y_train, hyperparams)
        train_pred = list(self.predict(x_train))
        validation_pred = list(self.predict(x_validation))
        if len(train_pred) != len(x_train):
            raise RuntimeError("predict output length must match x_train length")
        if len(validation_pred) != len(x_validation):
            raise RuntimeError("predict output length must match x_validation length")
        if self.task == "classification":
            return (
                y_train.astype(int).tolist(),
                [int(value) for value in train_pred],
                y_validation.astype(int).tolist(),
                [int(value) for value in validation_pred],
            )
        return (
            y_train.astype(float).tolist(),
            [float(value) for value in train_pred],
            y_validation.astype(float).tolist(),
            [float(value) for value in validation_pred],
        )

    def _sample_sandbox_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        sample_size = min(len(frame), SANDBOX_SAMPLE_ROWS)
        return frame.tail(sample_size).reset_index(drop=True)

    def _run_sandbox(self, train_time_limit_minutes: float) -> dict[str, object]:
        frame = self.prepare_data(min_rows=2)
        frame = self._sample_sandbox_frame(frame)
        prepared, feature_names = self.create_features(frame)
        prepared = prepared.reset_index(drop=True)
        train_frame, validation_frame = self._split_sandbox_data(prepared)
        hyperparams = self._build_single_hyperparameter_candidate()
        train_actual, train_pred, validation_actual, validation_pred = self._fit_predict_once(
            train_frame,
            validation_frame,
            feature_names,
            hyperparams,
        )
        metrics = self.evaluate(self.task, train_actual, train_pred, validation_actual, validation_pred)
        train_metrics = dict(metrics["train"])
        validation_metrics = dict(metrics["validation"])
        selected_hyperparams = dict(hyperparams)
        selected_hyperparams["training_size_days"] = 3
        train_metrics["selected_hyperparams"] = selected_hyperparams
        train_metrics["hyperparam_candidates_attempted"] = 1
        train_metrics["hyperparam_search_elapsed_minutes"] = 0.0
        train_metrics["train_time_limit_minutes"] = float(train_time_limit_minutes)
        self.best_hyperparams = dict(hyperparams)
        self.selected_training_size_days = 3
        self.train_metrics = train_metrics
        return {"train": train_metrics, "validation": validation_metrics}

    def train(
        self,
        frame: pd.DataFrame,
        feature_names: list[str],
        training_size_days: int,
        train_time_limit_minutes: float,
    ) -> None:
        if train_time_limit_minutes <= 0:
            raise RuntimeError("train_time_limit_minutes must be > 0")
        candidates = self._build_hyperparameter_candidates()
        best_params: dict[str, object] | None = None
        best_metrics: dict[str, object] | None = None
        best_score: float | None = None
        best_training_size_days: int | None = None
        errors: list[str] = []
        started_at = time.monotonic()
        time_limit_seconds = train_time_limit_minutes * 60.0
        attempted_count = 0
        for candidate in candidates:
            if attempted_count > 0 and (time.monotonic() - started_at) >= time_limit_seconds:
                break
            candidate_params = dict(candidate)
            candidate_training_size_days = int(candidate_params["training_size_days"])
            attempted_count += 1
            try:
                actual, predicted = self._walk_forward_predict(
                    frame=frame,
                    feature_names=feature_names,
                    training_size_days=candidate_training_size_days,
                    hyperparams=candidate_params,
                )
                metrics = self._metrics_from_predictions(actual, predicted)
                score = self._selection_score(metrics)
            except Exception as exc:
                errors.append(str(exc))
                continue
            if best_score is None or score > best_score:
                best_score = score
                best_params = candidate_params
                best_metrics = metrics
                best_training_size_days = candidate_training_size_days
        if best_params is None or best_metrics is None:
            joined = "; ".join(errors) if errors else "no valid candidates"
            raise RuntimeError(f"Hyperparameter search failed: {joined}")
        self.best_hyperparams = best_params
        self.selected_training_size_days = best_training_size_days or training_size_days
        train_metrics = dict(best_metrics)
        train_metrics["hyperparam_candidates_attempted"] = attempted_count
        train_metrics["hyperparam_search_elapsed_minutes"] = (time.monotonic() - started_at) / 60.0
        train_metrics["train_time_limit_minutes"] = float(train_time_limit_minutes)
        self.train_metrics = train_metrics

    def run(
        self,
        training_size_days: int = 30,
        train_time_limit_minutes: float = 5.0,
        execution_profile: str = "default",
    ) -> dict[str, object]:
        logger.info("Running model run_id=%s model_id=%s", self.run_id, self.model_id)
        if execution_profile == "sandbox":
            return self._run_sandbox(train_time_limit_minutes)
        if execution_profile != "default":
            raise RuntimeError(f"Unknown execution_profile: {execution_profile}")
        frame = self.prepare_data()
        prepared, feature_names = self.create_features(frame)
        train_frame, validation_frame = self.split_data(prepared, feature_names)
        self.train(train_frame, feature_names, training_size_days, train_time_limit_minutes)
        if self.train_metrics is None:
            raise RuntimeError("Training search did not produce metrics")
        resolved_training_size_days = int(self.selected_training_size_days or training_size_days)
        combined = pd.concat([train_frame, validation_frame], ignore_index=True)
        validation_start = validation_frame["timestamp"].min()
        validation_end = validation_frame["timestamp"].max()
        validation_actual, validation_pred = self._walk_forward_predict(
            combined,
            feature_names,
            resolved_training_size_days,
            self.best_hyperparams,
            test_range_start_ts=validation_start,
            test_range_end_ts=validation_end,
        )
        validation_metrics = self._metrics_from_predictions(validation_actual, validation_pred)
        train_metrics = dict(self.train_metrics)
        train_metrics["selected_hyperparams"] = dict(self.best_hyperparams)
        return {"train": train_metrics, "validation": validation_metrics}

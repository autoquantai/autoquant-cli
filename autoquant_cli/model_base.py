from __future__ import annotations

import logging
import random
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

from autoquant_cli.data import get_splits, load_dataset

logger = logging.getLogger(__name__)


def walk_forward(
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    training_size_days: int,
    test_size_days: int,
    first_test_at_start: bool = False,
):
    test_start_ts = start_ts if first_test_at_start else start_ts + pd.Timedelta(days=training_size_days)
    while test_start_ts < end_ts:
        test_end_ts = min(test_start_ts + pd.Timedelta(days=test_size_days), end_ts + pd.Timedelta(microseconds=1))
        train_start_ts = test_start_ts - pd.Timedelta(days=training_size_days)
        yield train_start_ts, test_start_ts, test_end_ts
        test_start_ts = test_end_ts


def _classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float | int | dict[str, float | int]]:
    true_values = [int(value) for value in y_true]
    pred_values = [int(value) for value in y_pred]
    if len(true_values) != len(pred_values):
        raise RuntimeError("y_true and y_pred must have equal length")
    if not true_values:
        raise RuntimeError("y_true and y_pred cannot be empty")
    report = classification_report(true_values, pred_values, output_dict=True, zero_division=0)
    class_one = report.get("1", {})
    macro_avg = report.get("macro avg", {})
    weighted_avg = report.get("weighted avg", {})
    n_samples = int(len(true_values))
    y_dist = float(sum(1 for value in true_values if value == 1) / n_samples)
    return {
        "n_samples": n_samples,
        "accuracy": float(report["accuracy"]),
        "precision": float(class_one.get("precision", 0.0)),
        "recall": float(class_one.get("recall", 0.0)),
        "f1": float(class_one.get("f1-score", 0.0)),
        "weighted_f1": float(weighted_avg.get("f1-score", 0.0)),
        "macro_f1": float(macro_avg.get("f1-score", 0.0)),
        "y_dist": y_dist,
        "report": report,
    }


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


def evaluate_predictions(
    task: str,
    train_actual: Sequence[float | int],
    train_pred: Sequence[float | int],
    validation_actual: Sequence[float | int],
    validation_pred: Sequence[float | int],
) -> dict[str, object]:
    if task == "classification":
        train_metrics = _classification_metrics([int(value) for value in train_actual], [int(value) for value in train_pred])
        validation_metrics = _classification_metrics(
            [int(value) for value in validation_actual],
            [int(value) for value in validation_pred],
        )
        return {"train": train_metrics, "validation": validation_metrics}
    if task == "regression":
        train_metrics = _regression_metrics([float(value) for value in train_actual], [float(value) for value in train_pred])
        validation_metrics = _regression_metrics(
            [float(value) for value in validation_actual],
            [float(value) for value in validation_pred],
        )
        return {"train": train_metrics, "validation": validation_metrics}
    raise RuntimeError("task must be classification or regression")


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

    def prepare_data(self) -> pd.DataFrame:
        frame = load_dataset(self.run_id)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["timestamp"]).reset_index(drop=True)
        if frame.empty:
            raise RuntimeError("Dataset is empty")
        min_ts = frame["timestamp"].min()
        max_ts = frame["timestamp"].max()
        if (max_ts - min_ts) < pd.Timedelta(days=30):
            raise RuntimeError("Dataset must span at least 30 days")
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

    def _get_attempted_window_count(
        self,
        frame: pd.DataFrame,
        training_size_days: int,
        test_size_days: int,
        first_test_at_start: bool = False,
    ) -> int:
        if frame.empty:
            return 0
        start_ts = frame["timestamp"].min()
        end_ts = frame["timestamp"].max()
        return sum(
            1
            for _ in walk_forward(
                start_ts=start_ts,
                end_ts=end_ts,
                training_size_days=training_size_days,
                test_size_days=test_size_days,
                first_test_at_start=first_test_at_start,
            )
        )

    def _enforce_min_windows(
        self,
        partition_name: str,
        frame: pd.DataFrame,
        training_size_days: int,
        test_size_days: int,
        min_windows: int,
        first_test_at_start: bool = False,
    ) -> None:
        attempted = self._get_attempted_window_count(
            frame=frame,
            training_size_days=training_size_days,
            test_size_days=test_size_days,
            first_test_at_start=first_test_at_start,
        )
        if attempted < min_windows:
            raise RuntimeError(
                f"{partition_name} requires at least {min_windows} walk-forward windows but got {attempted} "
                f"for training_size_days={training_size_days} test_size_days={test_size_days}"
            )

    def validate_model(
        self,
        train_frame: pd.DataFrame,
        validation_frame: pd.DataFrame,
        training_size_days: int,
        test_size_days: int,
    ) -> None:
        if training_size_days <= 0 or test_size_days <= 0:
            raise RuntimeError("training_size_days and test_size_days must be > 0")
        validation_start = validation_frame["timestamp"].min()
        lookback_start = validation_start - pd.Timedelta(days=training_size_days)
        if train_frame["timestamp"].min() > lookback_start:
            raise RuntimeError(
                f"Train partition does not provide enough lookback for validation: need data from {lookback_start} "
                f"but train starts at {train_frame['timestamp'].min()}"
            )
        self._enforce_min_windows("train", train_frame, training_size_days, test_size_days, 4)
        self._enforce_min_windows("validation", validation_frame, training_size_days, test_size_days, 2, True)

    def evaluate(
        self,
        train_actual: list[float | int],
        train_pred: list[float | int],
        validation_actual: list[float | int],
        validation_pred: list[float | int],
    ) -> dict[str, object]:
        return evaluate_predictions(self.task, train_actual, train_pred, validation_actual, validation_pred)

    def get_hyperparameter_candidates(self) -> list[dict[str, object]]:
        return [{}]

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
            return float(metrics["weighted_f1"])
        return float(metrics["r2"])

    def _walk_forward_predict(
        self,
        frame: pd.DataFrame,
        feature_names: list[str],
        training_size_days: int,
        test_size_days: int,
        hyperparams: dict[str, object],
        test_range_start_ts: pd.Timestamp | None = None,
        test_range_end_ts: pd.Timestamp | None = None,
    ) -> tuple[list[float | int], list[float | int]]:
        if frame.empty:
            raise RuntimeError("Partition is empty")
        if test_range_start_ts is not None and test_range_end_ts is not None:
            start_ts = test_range_start_ts
            end_ts = test_range_end_ts
            first_test_at_start = True
        else:
            start_ts = frame["timestamp"].min()
            end_ts = frame["timestamp"].max()
            first_test_at_start = False
        actual: list[float | int] = []
        predicted: list[float | int] = []
        for train_start_ts, test_start_ts, test_end_ts in walk_forward(
            start_ts=start_ts,
            end_ts=end_ts,
            training_size_days=training_size_days,
            test_size_days=test_size_days,
            first_test_at_start=first_test_at_start,
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
        paired = self.evaluate(actual, predicted, actual, predicted)
        metrics = paired["train"]
        if not isinstance(metrics, dict):
            raise RuntimeError("Invalid metrics payload from evaluate")
        return metrics

    def train(
        self,
        frame: pd.DataFrame,
        feature_names: list[str],
        training_size_days: int,
        test_size_days: int,
        train_time_limit_minutes: float,
    ) -> None:
        if train_time_limit_minutes <= 0:
            raise RuntimeError("train_time_limit_minutes must be > 0")
        candidates = self.get_hyperparameter_candidates() or [{}]
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
            candidate_params = dict(candidate or {})
            candidate_training_size_days = int(random.randint(15, max(15, training_size_days)))
            candidate_params["training_size_days"] = candidate_training_size_days
            attempted_count += 1
            try:
                actual, predicted = self._walk_forward_predict(
                    frame=frame,
                    feature_names=feature_names,
                    training_size_days=candidate_training_size_days,
                    test_size_days=test_size_days,
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
        test_size_days: int = 7,
        train_time_limit_minutes: float = 5.0,
    ) -> dict[str, object]:
        logger.info("Running model run_id=%s model_id=%s", self.run_id, self.model_id)
        frame = self.prepare_data()
        prepared, feature_names = self.create_features(frame)
        train_frame, validation_frame = self.split_data(prepared, feature_names)
        self.validate_model(train_frame, validation_frame, training_size_days, test_size_days)
        self.train(train_frame, feature_names, training_size_days, test_size_days, train_time_limit_minutes)
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
            test_size_days,
            self.best_hyperparams,
            test_range_start_ts=validation_start,
            test_range_end_ts=validation_end,
        )
        validation_metrics = self._metrics_from_predictions(validation_actual, validation_pred)
        train_metrics = dict(self.train_metrics)
        train_metrics["selected_hyperparams"] = dict(self.best_hyperparams)
        return {"train": train_metrics, "validation": validation_metrics}

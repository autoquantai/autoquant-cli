from __future__ import annotations

from datetime import date, timedelta

MINIMUM_EXPERIMENT_DATA_DAYS = 365


def _parse_date(value: str, field_name: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise RuntimeError(f"{field_name} must be a valid ISO date") from exc


def validate_experiment_dates(
    from_date: str,
    to_date: str,
    minimum_data_days: int = MINIMUM_EXPERIMENT_DATA_DAYS,
) -> None:
    parsed_from_date = _parse_date(from_date, "from_date")
    parsed_to_date = _parse_date(to_date, "to_date")
    if parsed_to_date < parsed_from_date:
        raise RuntimeError("to_date must be on or after from_date")
    if (parsed_to_date - parsed_from_date) < timedelta(days=minimum_data_days):
        raise RuntimeError(
            f"Experiment requires at least {minimum_data_days} days of data"
        )

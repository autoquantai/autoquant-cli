from __future__ import annotations

from dataclasses import dataclass

DEFAULT_CCXT_EXCHANGE = "binance"
DEFAULT_DATA_PROVIDER = "massive"
SUPPORTED_DATA_PROVIDERS = ("massive", "ccxt")


@dataclass(frozen=True)
class RunMarketConfig:
    input_ohlc_tickers: list[str]
    target_ticker: str
    data_provider: str
    ccxt_exchange: str | None

    @property
    def all_tickers(self) -> list[str]:
        return [self.target_ticker, *self.input_ohlc_tickers]


def _normalize_ticker(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise RuntimeError(f"{field_name} must not be empty")
    return normalized


def _normalize_ticker_list(values: list[str] | tuple[str, ...] | None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_value in values or []:
        ticker = _normalize_ticker(str(raw_value), "input_ohlc_tickers")
        if ticker in seen:
            continue
        seen.add(ticker)
        normalized.append(ticker)
    return normalized


def validate_run_market_config(
    input_ohlc_tickers: list[str] | tuple[str, ...] | None,
    target_ticker: str,
    data_provider: str = DEFAULT_DATA_PROVIDER,
    ccxt_exchange: str | None = None,
) -> RunMarketConfig:
    normalized_target_ticker = _normalize_ticker(target_ticker, "target_ticker")
    normalized_input_tickers = [
        ticker for ticker in _normalize_ticker_list(input_ohlc_tickers) if ticker != normalized_target_ticker
    ]
    normalized_data_provider = str(data_provider).strip().lower()
    if normalized_data_provider not in SUPPORTED_DATA_PROVIDERS:
        supported = ", ".join(SUPPORTED_DATA_PROVIDERS)
        raise RuntimeError(f"data_provider must be one of: {supported}")
    normalized_exchange = None
    if normalized_data_provider == "ccxt":
        normalized_exchange = str(ccxt_exchange or DEFAULT_CCXT_EXCHANGE).strip().lower()
        if not normalized_exchange:
            raise RuntimeError("ccxt_exchange must not be empty")
    elif ccxt_exchange:
        raise RuntimeError("ccxt_exchange is only valid when data_provider is ccxt")
    return RunMarketConfig(
        input_ohlc_tickers=normalized_input_tickers,
        target_ticker=normalized_target_ticker,
        data_provider=normalized_data_provider,
        ccxt_exchange=normalized_exchange,
    )

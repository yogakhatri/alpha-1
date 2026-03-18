from __future__ import annotations

"""Data quality guardrails used across pipeline stages."""

import pandas as pd


def assert_non_empty(df: pd.DataFrame, dataset_name: str) -> None:
    """Raise if dataframe is empty, with dataset-specific error text."""
    if df is None or df.empty:
        raise ValueError(f"{dataset_name} is empty.")


def latest_trading_day_lag(latest_date: pd.Timestamp, cfg_end_date: str) -> int:
    """Compute approximate NSE trading-day gap between latest row and config end date."""
    from src.core.calendar import nse_business_day_count
    end = pd.Timestamp(cfg_end_date).normalize()
    latest = pd.Timestamp(latest_date).normalize()
    if latest >= end:
        return 0
    return nse_business_day_count(latest, end)


def assert_fresh_enough(
    df: pd.DataFrame,
    cfg_end_date: str,
    max_stale_trading_days: int,
    dataset_name: str,
    date_col: str = "date",
) -> pd.Timestamp:
    """Validate freshness and return the latest date present in dataset."""
    assert_non_empty(df, dataset_name)
    if date_col not in df.columns:
        raise ValueError(f"{dataset_name} missing '{date_col}' column.")

    latest = pd.to_datetime(df[date_col]).max()
    lag = latest_trading_day_lag(latest, cfg_end_date)
    if lag > int(max_stale_trading_days):
        raise ValueError(
            f"{dataset_name} is stale by {lag} trading days "
            f"(latest={latest.date()}, cfg_end={cfg_end_date}, "
            f"allowed={max_stale_trading_days})."
        )
    return latest

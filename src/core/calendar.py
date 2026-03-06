from __future__ import annotations

"""Trading-day helpers used by data download and labeling logic."""

import pandas as pd


def trading_days_between(start: str, end: str) -> pd.DatetimeIndex:
    """Return business-day index between two dates.

    Note:
        This is a pragmatic approximation for NSE trading sessions. Exchange
        holidays are naturally handled by missing market data downstream.
    """
    # Minimal: business days. For v1 we accept that some NSE holidays will be included;
    # missing bhavcopy rows will naturally drop those days.
    return pd.bdate_range(start=start, end=end, tz=None)


def next_trading_day(d: pd.Timestamp, all_days: pd.DatetimeIndex) -> pd.Timestamp:
    """Return the next available date in the provided trading-day index."""
    pos = all_days.get_indexer([d], method="pad")[0]
    nxt = pos + 1
    if nxt >= len(all_days):
        raise ValueError("No next trading day in range")
    return all_days[nxt]

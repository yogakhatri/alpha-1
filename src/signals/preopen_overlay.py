from __future__ import annotations

"""Helpers for pre-open execution overlays."""

import pandas as pd


def open_price_for_execution(ohlcv: pd.DataFrame, trade_date: pd.Timestamp, ticker: str) -> float | None:
    """Return open price for ticker on trade date, or None if missing."""
    row = ohlcv[(ohlcv["ticker"] == ticker) & (ohlcv["date"] == trade_date)]
    if row.empty:
        return None
    return float(row.iloc[0]["Open"])

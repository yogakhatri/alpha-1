from __future__ import annotations

"""Merge raw sources into one canonical OHLCV table."""

import pandas as pd


def merge_primary_with_fallback(primary: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    """Merge primary and fallback OHLCV, prioritizing primary per ticker-date."""
    # primary expected: bhavcopy (date,ticker,OHLCV)
    if primary is None or primary.empty:
        return fallback.copy()

    p = primary.copy()
    p["date"] = pd.to_datetime(p["date"]).dt.tz_localize(None)
    p = p[["date", "ticker", "Open", "High", "Low", "Close", "Volume"]]
    p = p.sort_values(["ticker", "date"]).drop_duplicates(subset=["ticker", "date"], keep="last").copy()
    p["source_priority"] = 0

    if fallback is None or fallback.empty:
        return p.drop(columns=["source_priority"]).sort_values(["ticker", "date"]).reset_index(drop=True)

    f = fallback.copy()
    f["date"] = pd.to_datetime(f["date"]).dt.tz_localize(None)
    f = f[["date", "ticker", "Open", "High", "Low", "Close", "Volume"]]
    f = f.sort_values(["ticker", "date"]).drop_duplicates(subset=["ticker", "date"], keep="last").copy()
    f["source_priority"] = 1

    merged = pd.concat([p, f], ignore_index=True)
    merged = merged.sort_values(["ticker", "date", "source_priority"])
    merged = merged.drop_duplicates(subset=["ticker", "date"], keep="first")
    merged = merged.drop(columns=["source_priority"])
    return merged.reset_index(drop=True)

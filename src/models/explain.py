from __future__ import annotations

"""Feature-schema helpers shared with alpha generation."""

import pandas as pd


BASE_FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "returns_1d", "returns_5d", "RSI_14", "MACD", "EMA_20", "EMA_50", "ATR_14", "vol_z",
    "turnover", "turnover_med_20d",
]


def feature_schema_from_df(df: pd.DataFrame) -> list[str]:
    """Return allowed base feature names present in dataframe."""
    cols = [c for c in BASE_FEATURES if c in df.columns]
    return cols

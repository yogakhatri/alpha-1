from __future__ import annotations

"""Feature-table builder from normalized OHLCV input."""

import numpy as np
import pandas as pd

from src.core.config import AppConfig
from src.data_layer.symbol_master import load_tickers
from src.features.indicators import atr, ema, macd, rsi
from src.features.feature_validation import validate_features


def build_feature_table(cfg: AppConfig, ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Create per-ticker rolling feature set used by model and alpha formulas."""
    df = ohlcv.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    required = ["date", "ticker", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV missing required columns: {missing}. Have={list(df.columns)}")

    df = df[required].sort_values(["ticker", "date"]).drop_duplicates(subset=["ticker", "date"], keep="last").reset_index(drop=True)

    tickers = load_tickers(cfg.universe.tickers_file)
    if tickers:
        df = df[df["ticker"].isin(set(tickers))].copy()

    # Compute indicators per ticker so rolling windows never cross symbols.
    parts: list[pd.DataFrame] = []
    for t, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True).copy()

        close = g["Close"].astype(float)
        high = g["High"].astype(float)
        low = g["Low"].astype(float)
        vol = g["Volume"].astype(float)

        g["returns_1d"] = close.pct_change(1)
        g["returns_5d"] = close.pct_change(5)

        g["EMA_20"] = ema(close, 20)
        g["EMA_50"] = ema(close, 50)
        g["RSI_14"] = rsi(close, 14)
        g["MACD"] = macd(close)
        g["ATR_14"] = atr(high, low, close, 14)

        g["vol_z"] = (vol - vol.rolling(20).mean()) / (vol.rolling(20).std().replace(0.0, np.nan))
        g["turnover"] = (close * vol).astype(float)
        g["turnover_med_20d"] = g["turnover"].rolling(20).median()

        g["ticker"] = t  # ensure it exists
        parts.append(g)

    feats = pd.concat(parts, ignore_index=True)
    if "ticker" not in feats.columns:
        raise RuntimeError("BUG: ticker column missing after feature build")

    if cfg.universe.min_price > 0:
        feats = feats[feats["Close"] >= float(cfg.universe.min_price)].copy()
    if cfg.universe.min_median_turnover_20d > 0:
        feats = feats[feats["turnover_med_20d"] >= float(cfg.universe.min_median_turnover_20d)].copy()

    return validate_features(feats)

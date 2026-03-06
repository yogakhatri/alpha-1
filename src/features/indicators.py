from __future__ import annotations

"""Technical indicator primitives used during feature engineering."""

import numpy as np
import pandas as pd


def ema(s: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return s.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder-style exponential smoothing)."""
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0.0, np.nan))
    return 100.0 - (100.0 / (1.0 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD histogram-like series (MACD minus signal line)."""
    m = ema(close, fast) - ema(close, slow)
    sig = ema(m, signal)
    return m - sig  # MACD histogram-like


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range using exponential smoothing."""
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

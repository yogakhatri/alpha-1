from __future__ import annotations

"""Inference-window builder with explicit signal-timing semantics."""

from typing import Literal

import numpy as np
import pandas as pd


SignalTiming = Literal["PREOPEN_SAME_DAY", "EOD_NEXT_OPEN"]


def build_inference_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    lookback: int,
    signal_timing: SignalTiming,
    require_execution_day: bool = False,
) -> tuple[list[np.ndarray], list[str], list[pd.Timestamp], list[pd.Timestamp]]:
    """Build model input windows and aligned prediction/execution dates.

    The function centralizes timing logic so training, backtest, and signal
    generation remain consistent.
    """
    windows: list[np.ndarray] = []
    tickers: list[str] = []
    prediction_dates: list[pd.Timestamp] = []
    execution_dates: list[pd.Timestamp] = []

    for tkr, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date")
        arr = g[feature_cols].to_numpy(dtype=np.float32)
        d_arr = pd.to_datetime(g["date"]).to_numpy()
        n = len(g)

        if signal_timing == "PREOPEN_SAME_DAY":
            start = lookback
            stop = n
            for i in range(start, stop):
                windows.append(arr[i - lookback:i])
                tickers.append(tkr)
                prediction_dates.append(pd.Timestamp(d_arr[i]))
                execution_dates.append(pd.Timestamp(d_arr[i]))
            continue

        # EOD_NEXT_OPEN:
        # Use features up to date T (inclusive) and execute at Open(T+1).
        start = max(lookback - 1, 0)
        stop = n - 1 if require_execution_day else n
        for i in range(start, stop):
            windows.append(arr[i - lookback + 1:i + 1])
            tickers.append(tkr)
            prediction_dates.append(pd.Timestamp(d_arr[i]))
            if i + 1 < n:
                execution_dates.append(pd.Timestamp(d_arr[i + 1]))
            else:
                execution_dates.append(pd.NaT)

    return windows, tickers, prediction_dates, execution_dates

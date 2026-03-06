from __future__ import annotations

"""Barrier-label construction for supervised training/backtest alignment."""

import numpy as np
import pandas as pd

from src.core.config import AppConfig, pct_to_fraction


def add_label_column_barrier(cfg: AppConfig, df: pd.DataFrame) -> pd.DataFrame:
    """Attach binary outcome label per row using future-barrier execution logic.

    Label meaning:
    - `1`: target is reached before stop within horizon
    - `0`: stop hits first or trade times out
    - `NaN`: insufficient future data to resolve outcome
    """
    # Label at day T is defined by entry at Open(T+1); we attach label on row T for modeling.
    # We need future OHLC to compute outcome; this is used only for backtesting/training.
    g = df.sort_values(["ticker", "date"]).reset_index(drop=True).copy()
    horizon = max(1, int(cfg.label.horizon_days))
    pt_pct = pct_to_fraction(cfg.label.profit_take_pct)
    sl_pct = pct_to_fraction(cfg.label.stop_loss_pct)
    stop_first = cfg.label.same_day_both_hit_rule == "STOP_FIRST"

    labels = np.full(len(g), np.nan, dtype=float)

    for _, grp in g.groupby("ticker", sort=False):
        idx = grp.index.to_numpy()
        n = len(grp)
        if n == 0:
            continue

        op = grp["Open"].to_numpy(dtype=float)
        hi = grp["High"].to_numpy(dtype=float)
        lo = grp["Low"].to_numpy(dtype=float)

        entry = np.full(n, np.nan, dtype=float)
        entry[:-1] = op[1:]
        valid = (~np.isnan(entry)) & (entry > 0)
        target = entry * (1.0 + pt_pct)
        stop = entry * (1.0 - sl_pct)

        resolved = np.zeros(n, dtype=bool)
        out = np.zeros(n, dtype=float)  # default unresolved-valid outcome is stop/timeout => 0.0

        for day in range(1, horizon + 1):
            op_d = np.full(n, np.nan, dtype=float)
            hi_d = np.full(n, np.nan, dtype=float)
            lo_d = np.full(n, np.nan, dtype=float)

            if day < n:
                op_d[:-day] = op[day:]
                hi_d[:-day] = hi[day:]
                lo_d[:-day] = lo[day:]

            day_valid = (~np.isnan(op_d)) & (~np.isnan(hi_d)) & (~np.isnan(lo_d))
            valid &= day_valid
            pending = valid & (~resolved)
            if not np.any(pending):
                continue

            in_range = (op_d > stop) & (op_d < target)
            both = in_range & (hi_d >= target) & (lo_d <= stop)

            stop_hit = (op_d <= stop) | (in_range & (lo_d <= stop) & (~both)) | (both & stop_first)
            target_hit = (op_d >= target) | (in_range & (hi_d >= target) & (~both)) | (both & (not stop_first))

            do_stop = pending & stop_hit
            do_target = pending & (~stop_hit) & target_hit

            out[do_stop] = 0.0
            out[do_target] = 1.0
            resolved |= do_stop | do_target

        labels[idx[valid]] = out[valid]

    g["label"] = labels
    return g

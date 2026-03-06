from __future__ import annotations

"""Performance metrics for equity curves and trade-level outcomes."""

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    """Worst peak-to-trough decline of the equity curve."""
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())


def cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    """Compound annual growth rate from first to last equity point."""
    if len(equity) < 2:
        return 0.0
    total = equity.iloc[-1] / equity.iloc[0]
    years = (len(equity) - 1) / periods_per_year
    return float(total ** (1 / years) - 1) if years > 0 else 0.0


def profit_factor(trade_pnl: pd.Series) -> float:
    """Gross profit divided by gross loss."""
    gp = trade_pnl[trade_pnl > 0].sum()
    gl = -trade_pnl[trade_pnl < 0].sum()
    return float(gp / gl) if gl > 0 else float("inf")


def expectancy(trade_pnl: pd.Series) -> float:
    """Average profit/loss per trade."""
    return float(trade_pnl.mean()) if len(trade_pnl) else 0.0


def win_rate(trade_pnl: pd.Series) -> float:
    """Fraction of trades with positive PnL."""
    return float((trade_pnl > 0).mean()) if len(trade_pnl) else 0.0

from __future__ import annotations

"""Simple typed position model used by portfolio/backtest layers."""

from dataclasses import dataclass


@dataclass
class Position:
    """Represents one live long position in the simulator."""
    ticker: str
    entry_date: object
    entry_price: float
    shares: int
    stop: float
    target: float
    max_exit_date: object

from __future__ import annotations

"""Trading cost helpers used by entry/exit simulation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CostParams:
    """Container for one-side trading cost assumptions."""
    slippage_bps_per_side: float
    regulatory_bps_per_side: float
    groww_brokerage_enabled: bool = True


def groww_brokerage(trade_value_inr: float) -> float:
    """Approximate Groww brokerage for one order leg."""
    # Lower of ₹20 or 0.1% of trade value, minimum ₹5
    b = min(20.0, 0.001 * float(trade_value_inr))
    return max(5.0, b)


def bps_cost(trade_value_inr: float, bps: float) -> float:
    """Convert basis-points cost into currency amount for a trade value."""
    return float(trade_value_inr) * (bps / 10000.0)


def total_cost_one_side(trade_value_inr: float, params: CostParams) -> float:
    """Total one-side cost = bps costs + optional brokerage."""
    c = bps_cost(trade_value_inr, params.slippage_bps_per_side + params.regulatory_bps_per_side)
    if params.groww_brokerage_enabled:
        c += groww_brokerage(trade_value_inr)
    return c

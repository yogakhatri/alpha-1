from __future__ import annotations

"""Trading-day helpers used by data download and labeling logic."""

import pandas as pd
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday
from pandas.tseries.offsets import CustomBusinessDay


# Major NSE holidays (fixed-date; movable ones like Diwali are approximated)
# These cover Republic Day, Independence Day, Gandhi Jayanti, Christmas, etc.
# Movable holidays (Holi, Diwali, Eid, etc.) change yearly but skipping them
# is harmless — bhavcopy download will simply get a 404 and move on.
_NSE_FIXED_HOLIDAYS = [
    Holiday("Republic Day", month=1, day=26),
    Holiday("Independence Day", month=8, day=15),
    Holiday("Gandhi Jayanti", month=10, day=2),
    Holiday("Christmas", month=12, day=25),
    Holiday("May Day", month=5, day=1),
]


class NSEHolidayCalendar(AbstractHolidayCalendar):
    rules = _NSE_FIXED_HOLIDAYS


_NSE_BDAY = CustomBusinessDay(calendar=NSEHolidayCalendar())


def trading_days_between(start: str, end: str) -> pd.DatetimeIndex:
    """Return approximate NSE trading-day index between two dates.

    Uses weekday exclusion + known fixed NSE holidays. Movable holidays
    (Diwali, Holi, Eid) are not modeled — missing bhavcopy for those
    days is handled gracefully downstream.
    """
    return pd.date_range(start=start, end=end, freq=_NSE_BDAY)


def nse_business_day_count(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Count approximate NSE trading days between two dates."""
    if start >= end:
        return 0
    return len(pd.date_range(start=start + _NSE_BDAY, end=end, freq=_NSE_BDAY))


def next_trading_day(d: pd.Timestamp, all_days: pd.DatetimeIndex) -> pd.Timestamp:
    """Return the next available date in the provided trading-day index."""
    pos = all_days.get_indexer([d], method="pad")[0]
    nxt = pos + 1
    if nxt >= len(all_days):
        raise ValueError("No next trading day in range")
    return all_days[nxt]

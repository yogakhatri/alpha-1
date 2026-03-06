from __future__ import annotations

"""Feature-table sanitation helpers."""

import pandas as pd


def validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replace +/-inf with NA so later dropna logic can handle bad rows."""
    out = df.copy()
    out = out.replace([float("inf"), float("-inf")], pd.NA)
    return out

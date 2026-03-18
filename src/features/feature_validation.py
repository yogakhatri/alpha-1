from __future__ import annotations

"""Feature-table sanitation and quality checks."""

import numpy as np
import pandas as pd

from src.core.logging import get_logger

log = get_logger(__name__)

# Expected reasonable ranges for key features
_RANGE_CHECKS = {
    "RSI_14": (0, 100),
    "returns_1d": (-0.50, 0.50),
    "returns_5d": (-0.80, 0.80),
    "vol_z": (-10, 50),
}


def validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replace invalid values, check ranges, and log quality warnings."""
    out = df.copy()

    # Replace inf with NA
    out = out.replace([float("inf"), float("-inf")], pd.NA)

    # Range checks: clip extreme outliers and log warnings
    for col, (lo, hi) in _RANGE_CHECKS.items():
        if col not in out.columns:
            continue
        violations = ((out[col] < lo) | (out[col] > hi)).sum()
        if violations > 0:
            pct = 100.0 * violations / len(out)
            log.warning(
                "Feature %s: %d rows (%.2f%%) outside expected range [%s, %s]. Clipping.",
                col, violations, pct, lo, hi,
            )
            out[col] = out[col].clip(lower=lo, upper=hi)

    # Check for high null rates
    for col in out.columns:
        null_rate = out[col].isna().mean()
        if null_rate > 0.30:
            log.warning("Feature %s has %.1f%% null values", col, 100 * null_rate)

    # Log overall quality summary
    n_rows = len(out)
    n_nulls = int(out.isna().sum().sum())
    if n_rows > 0:
        log.info(
            "Feature validation: %d rows, %d total nulls (%.2f%% of cells)",
            n_rows, n_nulls, 100.0 * n_nulls / (n_rows * len(out.columns)),
        )

    return out

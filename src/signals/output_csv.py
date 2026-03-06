from __future__ import annotations

"""Small IO helper for writing signal tables to CSV."""

from pathlib import Path

import pandas as pd


def write_signals_csv(out_path: Path, signals: pd.DataFrame) -> Path:
    """Ensure parent folder exists, write CSV, and return written path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(out_path, index=False)
    return out_path

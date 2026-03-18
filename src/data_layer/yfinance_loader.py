from __future__ import annotations

"""Fallback data ingestion from yfinance with schema normalization.

This module is intentionally defensive because yfinance output schema can vary
across versions/environments (index naming, datetime column names, MultiIndex
columns, etc.). The loader normalizes everything into canonical columns:
`date, ticker, Open, High, Low, Close, Volume`.
"""

from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from src.core.config import AppConfig
from src.core.paths import RunPaths


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns into string names for predictable access."""
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            "_".join([str(x) for x in tup if str(x) not in {"", "None"}]).strip("_")
            for tup in out.columns.to_flat_index()
        ]
    else:
        out.columns = [str(c) for c in out.columns]
    return out


def _normalize_yfinance_frame(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Convert arbitrary yfinance dataframe shape into canonical OHLCV rows.

    Returns:
        pd.DataFrame: columns = date,ticker,Open,High,Low,Close,Volume
                      empty if required fields cannot be recovered.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    x = df.copy()
    if isinstance(x.index, pd.DatetimeIndex):
        idx_name = str(x.index.name) if x.index.name else "date"
        x = x.reset_index()
        if idx_name in x.columns:
            x = x.rename(columns={idx_name: "date"})
        elif "index" in x.columns:
            x = x.rename(columns={"index": "date"})
    x = _flatten_columns(x)

    # Normalize date column name.
    lower_to_col = {str(c).lower(): c for c in x.columns}
    if "date" in lower_to_col:
        x = x.rename(columns={lower_to_col["date"]: "date"})
    elif "datetime" in lower_to_col:
        x = x.rename(columns={lower_to_col["datetime"]: "date"})
    elif "index" in lower_to_col:
        x = x.rename(columns={lower_to_col["index"]: "date"})
    else:
        # Last-resort: pick first datetime-like column.
        for c in x.columns:
            if pd.api.types.is_datetime64_any_dtype(x[c]):
                x = x.rename(columns={c: "date"})
                break

    if "date" not in x.columns:
        return pd.DataFrame()

    def pick_col(base: str) -> str | None:
        """Pick column for OHLCV base name from exact or prefixed variants."""
        b = base.lower()
        exact = {str(c).lower(): c for c in x.columns}
        if b in exact:
            return exact[b]

        # Handles flattened names like Open_RELIANCE.NS / Open_RELIANCE.NS_...
        for c in x.columns:
            lc = str(c).lower()
            if lc.startswith(b + "_"):
                return c
        return None

    open_col = pick_col("Open")
    high_col = pick_col("High")
    low_col = pick_col("Low")
    close_col = pick_col("Close")
    volume_col = pick_col("Volume")

    if open_col is None or high_col is None or low_col is None or close_col is None:
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(x["date"], errors="coerce").dt.tz_localize(None),
            "ticker": ticker,
            "Open": pd.to_numeric(x[open_col], errors="coerce"),
            "High": pd.to_numeric(x[high_col], errors="coerce"),
            "Low": pd.to_numeric(x[low_col], errors="coerce"),
            "Close": pd.to_numeric(x[close_col], errors="coerce"),
            "Volume": pd.to_numeric(x[volume_col], errors="coerce") if volume_col else 0.0,
        }
    )
    out["Volume"] = out["Volume"].fillna(0.0)
    out = out.dropna(subset=["date", "Open", "High", "Low", "Close"])
    return out.reset_index(drop=True)


def download_yfinance_range(cfg: AppConfig, paths: RunPaths, tickers: list[str]) -> None:
    """Download per-ticker yfinance data and store normalized parquet files."""
    out_dir = paths.raw / "yfinance"
    out_dir.mkdir(parents=True, exist_ok=True)

    for t in tqdm(tickers, desc="yfinance"):
        out_p = out_dir / f"{t}.parquet"
        if out_p.exists():
            continue

        raw = yf.download(
            tickers=t,
            start=cfg.data.start_date,
            end=cfg.data.end_date,
            auto_adjust=True,
            progress=False,
        )
        norm = _normalize_yfinance_frame(raw, ticker=t)
        if norm.empty:
            continue
        norm.to_parquet(out_p, index=False)


def load_yfinance_ohlcv(cfg: AppConfig, paths: RunPaths) -> pd.DataFrame:
    """Load cached yfinance files and normalize legacy schema variants."""
    in_dir = paths.raw / "yfinance"
    if not in_dir.exists():
        return pd.DataFrame()

    rows = []
    for p in sorted(in_dir.glob("*.parquet")):
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        if df.empty:
            continue

        ticker = p.stem
        norm = _normalize_yfinance_frame(df, ticker=ticker)
        if norm.empty:
            continue
        rows.append(norm)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


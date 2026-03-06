from __future__ import annotations

"""Monitoring report generation for post-training quality/drift tracking."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.config import AppConfig
from src.core.paths import RunPaths


def save_monitoring_reports(cfg: AppConfig, paths: RunPaths, oos_df: pd.DataFrame) -> None:
    """Write daily and summary monitoring artifacts from OOS predictions."""
    if oos_df.empty:
        pd.DataFrame().to_csv(paths.monitoring_daily_path(), index=False)
        paths.monitoring_summary_path().write_text(json.dumps({"empty": True}, indent=2))
        return

    by_day = oos_df.groupby("date", as_index=False).agg(
        n_candidates=("ticker", "size"),
        base_rate=("label", "mean"),
    )
    top1 = (
        oos_df.sort_values(["date", "calibrated_prob"], ascending=[True, False])
        .groupby("date", as_index=False)
        .head(1)[["date", "label", "calibrated_prob"]]
        .rename(columns={"label": "top1_hit", "calibrated_prob": "top1_prob"})
    )
    top2 = (
        oos_df.sort_values(["date", "calibrated_prob"], ascending=[True, False])
        .groupby("date", as_index=False)
        .head(2)
        .groupby("date", as_index=False)
        .agg(
            top2_any_hit=("label", "max"),
            top2_pick_precision=("label", "mean"),
        )
    )

    daily = by_day.merge(top1, on="date", how="left").merge(top2, on="date", how="left")
    daily = daily.sort_values("date").reset_index(drop=True)

    short_w = int(cfg.monitoring.rolling_short_days)
    long_w = int(cfg.monitoring.rolling_long_days)
    drift_drop = float(cfg.monitoring.drift_alert_drop)

    daily["roll_top1_short"] = daily["top1_hit"].rolling(short_w, min_periods=short_w).mean()
    daily["roll_top1_long"] = daily["top1_hit"].rolling(long_w, min_periods=long_w).mean()
    daily["roll_top2_any_short"] = daily["top2_any_hit"].rolling(short_w, min_periods=short_w).mean()
    daily["drift_alert"] = (
        daily["roll_top1_short"] < (daily["roll_top1_long"] - drift_drop)
    ).fillna(False)

    daily.to_csv(paths.monitoring_daily_path(), index=False)

    summary = {
        "days": int(daily["date"].nunique()),
        "overall_top1_precision": float(daily["top1_hit"].mean()),
        "overall_top2_any_hit": float(daily["top2_any_hit"].mean()),
        "overall_base_rate": float(daily["base_rate"].mean()),
        "latest_roll_top1_short": float(daily["roll_top1_short"].dropna().iloc[-1]) if daily["roll_top1_short"].notna().any() else None,
        "latest_roll_top1_long": float(daily["roll_top1_long"].dropna().iloc[-1]) if daily["roll_top1_long"].notna().any() else None,
        "drift_alert_days": int(daily["drift_alert"].sum()),
    }
    paths.monitoring_summary_path().write_text(json.dumps(summary, indent=2))

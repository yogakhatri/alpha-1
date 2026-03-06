from __future__ import annotations

"""Probability calibration and threshold selection utilities."""
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def fit_platt_calibrator(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Fit logistic (Platt) calibrator mapping raw scores to probabilities."""
    x = np.asarray(scores, dtype=float).reshape(-1, 1)
    y = np.asarray(labels, dtype=int).reshape(-1)
    y = np.clip(y, 0, 1)

    # Degenerate data fallback.
    if len(np.unique(y)) < 2:
        return {"method": "identity"}

    lr = LogisticRegression(solver="lbfgs", max_iter=1000)
    lr.fit(x, y)
    return {
        "method": "platt",
        "coef": float(lr.coef_[0, 0]),
        "intercept": float(lr.intercept_[0]),
    }


def apply_calibration(scores: np.ndarray, calibration: dict[str, Any] | None) -> np.ndarray:
    """Apply calibration object to raw model scores."""
    arr = np.asarray(scores, dtype=float)
    if not calibration:
        return arr
    method = calibration.get("method", "identity")
    if method == "identity":
        return arr
    if method != "platt":
        return arr

    coef = float(calibration.get("coef", 1.0))
    intercept = float(calibration.get("intercept", 0.0))
    z = coef * arr + intercept
    # numerically stable sigmoid
    out = np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
    return np.clip(out, 1e-6, 1 - 1e-6)


def _pick_threshold(
    top_df: pd.DataFrame,
    score_col: str,
    target_precision: float,
    min_days: int,
) -> dict[str, float]:
    """Choose score threshold balancing precision target and coverage days."""
    if top_df.empty:
        return {"threshold": 1.0, "days": 0, "precision": 0.0}

    candidates = np.quantile(top_df[score_col], np.linspace(0.0, 0.98, 60))
    records: list[tuple[float, int, float]] = []
    for th in np.unique(candidates):
        s = top_df[top_df[score_col] >= float(th)]
        days = int(s["date"].nunique())
        if days == 0:
            continue
        prec = float(s["label"].mean())
        records.append((float(th), days, prec))

    if not records:
        return {"threshold": 1.0, "days": 0, "precision": 0.0}

    feasible = [r for r in records if r[1] >= min_days and r[2] >= target_precision]
    if feasible:
        # among feasible, maximize days first, then precision.
        feasible.sort(key=lambda x: (x[1], x[2], -x[0]), reverse=True)
        th, days, prec = feasible[0]
        return {"threshold": th, "days": days, "precision": prec}

    # fallback: maximize precision with at least min_days, else best precision overall
    at_least_days = [r for r in records if r[1] >= min_days]
    pool = at_least_days if at_least_days else records
    pool.sort(key=lambda x: (x[2], x[1], -x[0]), reverse=True)
    th, days, prec = pool[0]
    return {"threshold": th, "days": days, "precision": prec}


def build_thresholds_from_oos(
    oos_df: pd.DataFrame,
    score_col: str,
    target_top1_precision: float,
    min_days_for_threshold: int,
) -> dict[str, float]:
    """Derive top-1 threshold from out-of-sample prediction table."""
    if oos_df.empty:
        return {"top1_threshold": 1.0, "top1_days": 0, "top1_precision": 0.0}

    top1 = (
        oos_df.sort_values(["date", score_col], ascending=[True, False])
        .groupby("date", as_index=False)
        .head(1)
    )
    chosen = _pick_threshold(
        top_df=top1,
        score_col=score_col,
        target_precision=target_top1_precision,
        min_days=min_days_for_threshold,
    )
    return {
        "top1_threshold": float(chosen["threshold"]),
        "top1_days": int(chosen["days"]),
        "top1_precision": float(chosen["precision"]),
    }

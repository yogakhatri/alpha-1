"""Daily signal generation for the decision-support workflow.

This module computes model scores for each eligible ticker, applies calibration
and decision filters, and writes both:
1) a date-stamped signals file (immutable historical snapshot), and
2) a stable latest file for consumers that expect a fixed path.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.core.config import AppConfig, pct_to_fraction
from src.core.paths import RunPaths
from src.models.calibration import apply_calibration
from src.models.windowing import build_inference_windows


def generate_daily_signals(
    cfg: AppConfig,
    paths: RunPaths,
    feats: pd.DataFrame,
    ohlcv: pd.DataFrame,
    alpha_lib,
    model_bundle,
) -> str:
    """Generate and persist the final daily stock picks.

    Returns:
        str: path to the date-stamped signal file (signals_YYYYMMDD.csv).
    """
    from src.core.logging import get_logger
    log = get_logger(__name__)
    log.info("Preparing data for daily signals...")

    model = model_bundle.model
    scaler = model_bundle.scaler
    feature_cols = model_bundle.feature_cols
    lookback = int(model_bundle.lookback)
    signal_timing = getattr(model_bundle, "signal_timing", cfg.execution.signal_timing)
    if signal_timing != cfg.execution.signal_timing:
        log.warning(
            "Config signal_timing=%s differs from model timing=%s. Using model timing.",
            cfg.execution.signal_timing,
            signal_timing,
        )

    # ---------------------------------------------------------------
    # 1) Normalize to flat columns: ticker, date
    # ---------------------------------------------------------------
    df = feats.copy()

    if isinstance(df.index, pd.MultiIndex):
        idx_names = list(df.index.names)
        if "ticker" in idx_names or "date" in idx_names:
            df = df.reset_index()
    else:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "date"})

    if "ticker" not in df.columns and "Ticker" in df.columns:
        df = df.rename(columns={"Ticker": "ticker"})
    if "date" not in df.columns and "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})

    if "ticker" not in df.columns:
        raise ValueError(
            f"'ticker' column not found in feats. Available: {list(df.columns)[:30]}"
        )
    if "date" not in df.columns:
        raise ValueError(
            f"'date' column not found in feats. Available: {list(df.columns)[:30]}"
        )

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    tickers = paths.load_tickers(cfg.universe.tickers_file)
    if tickers:
        df = df[df["ticker"].isin(set(tickers))].copy()
    if cfg.universe.min_price > 0 and "Close" in df.columns:
        df = df[df["Close"] >= float(cfg.universe.min_price)].copy()
    if cfg.universe.min_median_turnover_20d > 0 and "turnover_med_20d" in df.columns:
        df = df[df["turnover_med_20d"] >= float(cfg.universe.min_median_turnover_20d)].copy()

    # ---------------------------------------------------------------
    # 2) Slice a rolling buffer (rolling funcs need history)
    # ---------------------------------------------------------------
    buffer_days = lookback * 2
    unique_dates = np.sort(df["date"].dropna().unique())
    if len(unique_dates) == 0:
        raise ValueError("No dates found in feats.")

    latest_dates = set(unique_dates[-buffer_days:])
    latest_df = df[df["date"].isin(latest_dates)].copy().reset_index(drop=True)

    # ---------------------------------------------------------------
    # 3) Compute LLM alphas dynamically on latest_df
    #    FIX: to_series() ensures numpy arrays from intermediate ops
    #    (e.g. np.log(Close)) are re-wrapped as pd.Series before any
    #    groupby-based rolling helper is called — preventing the
    #    "'numpy.ndarray' object has no attribute 'groupby'" warnings.
    # ---------------------------------------------------------------
    if alpha_lib and hasattr(alpha_lib, "formulas") and alpha_lib.formulas:
        log.info("Computing LLM alphas dynamically for inference...")

        def to_series(x):
            """Re-wrap a numpy array as a Series aligned to latest_df."""
            if isinstance(x, np.ndarray):
                return pd.Series(x, index=latest_df.index)
            return x

        def apply_grouped(s, func):
            s = to_series(s)
            return s.groupby(latest_df["ticker"]).transform(func)

        def safe_div(x, y):
            x, y = to_series(x), to_series(y)
            return np.where(y == 0, 0.0, x / y)

        def zscore(x, w):
            x = to_series(x)
            mean = apply_grouped(x, lambda s: s.rolling(w).mean())
            std  = apply_grouped(x, lambda s: s.rolling(w).std())
            return (x - mean) / np.where(std == 0, 1e-9, std)

        def delta(x, p):
            return apply_grouped(x, lambda s: s.diff(p))

        def rolling_mean(x, w):
            return apply_grouped(x, lambda s: s.rolling(w).mean())

        def rolling_std(x, w):
            return apply_grouped(x, lambda s: s.rolling(w).std())

        def rolling_min(x, w):
            return apply_grouped(x, lambda s: s.rolling(w).min())

        def rolling_max(x, w):
            return apply_grouped(x, lambda s: s.rolling(w).max())

        def sign(x):
            return np.sign(to_series(x))

        def clip(x, lower, upper):
            return np.clip(to_series(x), lower, upper)

        def log1p(x):
            return np.log1p(to_series(x))

        def shift(x, d):
            return apply_grouped(x, lambda s: s.shift(d))

        def ewm_mean(x, span):
            return apply_grouped(x, lambda s: s.ewm(span=span).mean())

        eval_globals = {
            "safe_div":    safe_div,
            "zscore":      zscore,
            "delta":       delta,
            "rolling_mean": rolling_mean,
            "rolling_std": rolling_std,
            "rolling_min": rolling_min,
            "rolling_max": rolling_max,
            "sign":        sign,
            "abs":         abs,
            "np":          np,
            "clip":        clip,
            "log1p":       log1p,
            "shift":       shift,
            "ewm_mean":    ewm_mean,
        }

        for alpha_name, formula in alpha_lib.formulas.items():
            if alpha_name in latest_df.columns:
                continue  # already present, skip

            eval_locals = {
                col: latest_df[col]
                for col in latest_df.columns
                if col not in ("ticker", "date")
            }

            # Explicitly expose OHLCV with canonical casing
            for canonical in ("Open", "High", "Low", "Close", "Volume"):
                if canonical in latest_df.columns:
                    eval_locals[canonical] = latest_df[canonical]

            try:
                result = eval(formula, eval_globals, eval_locals)
                latest_df[alpha_name] = to_series(result).values
            except Exception as e:
                log.warning(
                    f"Alpha {alpha_name} failed to compute: {e}. Filling with 0.0"
                )
                latest_df[alpha_name] = 0.0

    # ---------------------------------------------------------------
    # 4) Drop NAs, scale, build ONE window per ticker (tail=lookback)
    # ---------------------------------------------------------------
    available_feat_cols = [c for c in feature_cols if c in latest_df.columns]
    missing = [c for c in feature_cols if c not in latest_df.columns]
    if missing:
        log.warning(f"Missing feature columns (filling with 0): {missing}")
        for c in missing:
            latest_df[c] = 0.0
        available_feat_cols = feature_cols

    latest_df = latest_df.dropna(subset=available_feat_cols).copy()
    if latest_df.empty:
        raise ValueError(
            "No rows left after dropna(feature_cols). "
            "Increase buffer_days or check feature pipeline."
        )

    latest_df[available_feat_cols] = scaler.transform(latest_df[available_feat_cols])

    windows_all, tickers_all, pred_dates_all, exec_dates_all = build_inference_windows(
        df=latest_df,
        feature_cols=available_feat_cols,
        lookback=lookback,
        signal_timing=signal_timing,
        require_execution_day=False,
    )
    if len(windows_all) == 0:
        raise ValueError(
            "No tickers had enough clean history for an inference window. "
            f"Lookback={lookback}, buffer_days={buffer_days}."
        )

    meta = pd.DataFrame(
        {
            "idx": np.arange(len(windows_all)),
            "ticker": tickers_all,
            "prediction_date": pd.to_datetime(pred_dates_all),
            "execution_date": pd.to_datetime(exec_dates_all),
        }
    )
    meta = meta.sort_values(["ticker", "prediction_date"]).groupby("ticker", as_index=False).tail(1)
    sel_idx = meta["idx"].to_numpy(dtype=int)

    windows = [windows_all[i] for i in sel_idx]
    tickers = meta["ticker"].tolist()
    prediction_dates = meta["prediction_date"].tolist()
    execution_dates = meta["execution_date"].tolist()

    if len(windows) == 0:
        raise ValueError(
            "No tickers had enough clean history for a lookback window. "
            f"Lookback={lookback}, buffer_days={buffer_days}. "
            "Try increasing buffer_days or lowering lookback in config."
        )

    X  = torch.tensor(np.stack(windows), dtype=torch.float32)
    ds = TensorDataset(X, torch.zeros((X.shape[0], 1), dtype=torch.float32))
    loader = DataLoader(ds, batch_size=cfg.model.batch_size, shuffle=False)

    # ---------------------------------------------------------------
    # 5) Inference
    # ---------------------------------------------------------------
    device = torch.device("cpu")
    if cfg.model.use_mps_if_available and torch.backends.mps.is_available():
        device = torch.device("mps")

    model = model.to(device)
    model.eval()

    preds = []
    with torch.no_grad():
        for bx, _ in loader:
            bx = bx.to(device)
            out  = model(bx)
            probs = torch.sigmoid(out).detach().cpu().numpy().reshape(-1)
            preds.extend(probs.tolist())

    calibration = paths.load_model_calibration()
    calibrated_probs = apply_calibration(np.asarray(preds, dtype=float), calibration)
    tp_pct = pct_to_fraction(cfg.label.profit_take_pct)
    sl_pct = pct_to_fraction(cfg.label.stop_loss_pct)
    slot_size = float(cfg.portfolio.capital_inr) * float(cfg.portfolio.slot_capital_fraction)
    est_cost_frac = 2.0 * (
        float(cfg.costs.slippage_bps_per_side) + float(cfg.costs.regulatory_bps_per_side)
    ) / 10000.0
    if cfg.costs.groww_brokerage.enabled and slot_size > 0:
        est_cost_frac += (2.0 * 20.0) / slot_size
    expected_value_pct = calibrated_probs * tp_pct - (1.0 - calibrated_probs) * sl_pct - est_cost_frac

    # ---------------------------------------------------------------
    # 6) Rank and save
    # ---------------------------------------------------------------
    results_df = (
        pd.DataFrame(
            {
                "ticker": tickers,
                "prediction_date": prediction_dates,
                "execution_date": execution_dates,
                "score": preds,
                "calibrated_prob": calibrated_probs,
                "expected_value_pct": expected_value_pct,
            }
        )
    )

    min_prob = float(cfg.decision.min_calibrated_prob)
    if calibration and cfg.decision.no_trade_if_below_threshold:
        min_prob = max(min_prob, float(calibration.get("top1_threshold", 0.0)))

    filtered = results_df[
        (results_df["score"] >= float(cfg.decision.min_score_threshold))
        & (results_df["calibrated_prob"] >= min_prob)
        & (results_df["expected_value_pct"] >= float(cfg.decision.min_expected_value_pct))
    ].copy()
    rank_col = "expected_value_pct" if cfg.decision.rank_by == "expected_value" else "calibrated_prob"
    max_picks = max(0, int(cfg.decision.max_picks_per_day))

    filtered = (
        filtered.sort_values(rank_col, ascending=False)
        .head(max_picks)
        .reset_index(drop=True)
    )
    if filtered.empty:
        filtered = results_df.sort_values(rank_col, ascending=False).head(0).copy()

    # Create a deterministic date key for this signal batch.
    # Prefer execution date if available, else prediction date, else "today"
    # in project timezone. This guarantees unique per-date filenames.
    if not filtered.empty:
        date_source = filtered["execution_date"].dropna()
        if date_source.empty:
            date_source = filtered["prediction_date"].dropna()
    else:
        date_source = results_df["execution_date"].dropna()
        if date_source.empty:
            date_source = results_df["prediction_date"].dropna()
    if date_source.empty:
        signal_date = pd.Timestamp.now(tz=cfg.project.timezone).normalize().tz_localize(None)
    else:
        signal_date = pd.to_datetime(date_source).max()

    dated_csv = paths.signals_csv_path_for_date(signal_date)
    latest_csv = paths.signals_csv_path()

    # Write both files:
    # - dated snapshot for historical traceability
    # - stable latest file for downstream consumers
    filtered.to_csv(str(dated_csv), index=False)
    filtered.to_csv(str(latest_csv), index=False)

    log.info("Signals saved (dated): %s", dated_csv)
    log.info("Signals saved (latest): %s", latest_csv)
    if filtered.empty:
        log.info("No-trade day: no ticker passed decision thresholds.")
    else:
        log.info(f"\n{filtered.to_string()}")
    return str(dated_csv)

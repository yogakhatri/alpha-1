from __future__ import annotations
"""Walk-forward model training, OOS scoring, calibration, and monitoring."""

import copy
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.backtest.simulator import add_label_column_barrier
from src.core.config import AppConfig
from src.core.device import (
    inference_context,
    log_device_info,
    make_amp_context,
    optimal_batch_size,
    optimal_num_workers,
    pin_memory_for,
    resolve_device,
    try_compile,
    tune_for_device,
)
from src.core.paths import RunPaths
from src.core.seed import seed_everything
from src.models.calibration import apply_calibration, build_thresholds_from_oos, fit_platt_calibrator
from src.models.monitoring import save_monitoring_reports
from src.models.transformer import TimeSeriesTransformer

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ---------------------------------------------------------------------------
# Device helpers (delegated to src.core.device)
# ---------------------------------------------------------------------------

def get_device(cfg: AppConfig) -> torch.device:
    """Resolve training device: CUDA > MPS > CPU."""
    device = resolve_device(
        prefer_cuda=getattr(cfg.model, "use_cuda_if_available", True),
        prefer_mps=getattr(cfg.model, "use_mps_if_available", True),
    )
    tune_for_device(device)
    return device


# ---------------------------------------------------------------------------
# Model bundle
# ---------------------------------------------------------------------------

class ModelBundle:
    """Serializable container persisted after training."""
    def __init__(self, model, scaler, feature_cols, lookback, signal_timing: str = "EOD_NEXT_OPEN"):
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.lookback = lookback
        self.signal_timing = signal_timing


# ---------------------------------------------------------------------------
# Walk-forward splits
# ---------------------------------------------------------------------------

def _build_walk_forward_splits(dates: np.ndarray, cfg: AppConfig) -> list[dict[str, pd.Timestamp]]:
    """Create rolling train/val/test date boundaries from unique dates."""
    train_days = int(cfg.backtest.walk_forward.train_days)
    val_days   = int(cfg.backtest.walk_forward.val_days)
    test_days  = int(cfg.backtest.walk_forward.test_days)
    step       = int(cfg.backtest.walk_forward.step_days)
    total      = train_days + val_days + test_days
    if len(dates) < total:
        return []

    splits: list[dict[str, pd.Timestamp]] = []
    for s in range(0, len(dates) - total + 1, step):
        tr = dates[s : s + train_days]
        va = dates[s + train_days : s + train_days + val_days]
        te = dates[s + train_days + val_days : s + total]
        splits.append({
            "train_start": pd.Timestamp(tr[0]),
            "train_end":   pd.Timestamp(tr[-1]),
            "val_start":   pd.Timestamp(va[0]),
            "val_end":     pd.Timestamp(va[-1]),
            "test_start":  pd.Timestamp(te[0]),
            "test_end":    pd.Timestamp(te[-1]),
        })

    max_folds = cfg.backtest.walk_forward.max_folds
    if max_folds is not None and len(splits) > int(max_folds):
        splits = splits[-int(max_folds):]
    return splits


# ---------------------------------------------------------------------------
# Window building
# ---------------------------------------------------------------------------

def _build_labeled_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    lookback: int,
    signal_timing: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray, list[str], list[pd.Timestamp]]:
    """Build supervised windows for a date range using selected timing mode."""
    windows: list[np.ndarray] = []
    labels:  list[float]      = []
    tickers: list[str]        = []
    dates:   list[pd.Timestamp] = []

    for tkr, g in df.groupby("ticker", sort=False):
        g   = g.sort_values("date")
        arr = g[feature_cols].to_numpy(dtype=np.float32)
        y   = g["label"].to_numpy(dtype=np.float32)
        d   = pd.to_datetime(g["date"]).to_numpy()
        n   = len(g)

        if signal_timing == "PREOPEN_SAME_DAY":
            for i in range(lookback, n):
                dt = pd.Timestamp(d[i])
                if dt < start_date or dt > end_date:
                    continue
                if np.isnan(y[i]):
                    continue
                x = arr[i - lookback : i]
                if np.isnan(x).any():
                    continue
                windows.append(x); labels.append(float(y[i]))
                tickers.append(tkr); dates.append(dt)
            continue

        # EOD_NEXT_OPEN
        for i in range(max(lookback - 1, 0), n):
            dt = pd.Timestamp(d[i])
            if dt < start_date or dt > end_date:
                continue
            if np.isnan(y[i]):
                continue
            x = arr[i - lookback + 1 : i + 1]
            if np.isnan(x).any():
                continue
            windows.append(x); labels.append(float(y[i]))
            tickers.append(tkr); dates.append(dt)

    if not windows:
        return (
            np.empty((0, lookback, len(feature_cols)), dtype=np.float32),
            np.empty((0,), dtype=np.float32), [], [],
        )
    return np.stack(windows).astype(np.float32), np.asarray(labels, dtype=np.float32), tickers, dates


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def _to_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    device: torch.device | None = None,
) -> DataLoader:
    """Wrap NumPy arrays into a torch DataLoader with device-optimal settings."""
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y.reshape(-1, 1), dtype=torch.float32),
    )
    nw = optimal_num_workers(device) if device is not None else 0
    pm = pin_memory_for(device) if device is not None else False
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=nw,
        pin_memory=pm,
        persistent_workers=(nw > 0),
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _predict_scores(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int | None = None,
) -> np.ndarray:
    """Run batched inference and return sigmoid scores."""
    if len(X) == 0:
        return np.empty((0,), dtype=np.float32)

    pm = pin_memory_for(device)
    autocast_fn, _ = make_amp_context(device)
    if batch_size is None:
        batch_size = optimal_batch_size(device)["inference"]
    nw = optimal_num_workers(device)

    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=pm,
        persistent_workers=(nw > 0),
    )
    out: list[float] = []
    model.eval()
    with inference_context():
        for (bx,) in loader:
            bx = bx.to(device, non_blocking=pm)
            with autocast_fn():
                logits = model(bx)
            probs = torch.sigmoid(logits).float().cpu().numpy().reshape(-1)
            out.extend(probs.tolist())
    return np.asarray(out, dtype=np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _daily_topk_precision(dates, scores, labels, k):
    if len(scores) == 0:
        return 0.0
    d = pd.DataFrame({"date": pd.to_datetime(dates), "score": scores, "label": labels})
    top = d.sort_values(["date", "score"], ascending=[True, False]).groupby("date", as_index=False).head(k)
    return float(top["label"].mean()) if len(top) else 0.0


def _compute_pos_weight(y: np.ndarray) -> float:
    pos = float((y > 0.5).sum())
    neg = float((y <= 0.5).sum())
    if pos <= 0:
        return 1.0
    return float(np.clip(neg / pos, 1.0, 20.0))


# ---------------------------------------------------------------------------
# Fold training
# ---------------------------------------------------------------------------

def _train_fold_model(
    cfg: AppConfig,
    device: torch.device,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_dates: list[pd.Timestamp],
) -> tuple[nn.Module, dict[str, float]]:
    """Train one fold model with early stopping on rank-aware validation metric."""
    model = TimeSeriesTransformer(
        num_features=X_train.shape[-1],
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        dropout=cfg.model.dropout,
        num_classes=1,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.model.lr,
        weight_decay=cfg.model.weight_decay,
    )

    pos_weight = torch.tensor([_compute_pos_weight(y_train)], dtype=torch.float32, device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    pm          = pin_memory_for(device)
    autocast_fn, scaler = make_amp_context(device)
    bs          = optimal_batch_size(device, base=cfg.model.batch_size)

    loader     = _to_loader(X_train, y_train, bs["train"],     shuffle=True,  device=device)
    val_loader = _to_loader(X_val,   y_val,   bs["inference"], shuffle=False, device=device)

    best_metric = float("-inf")
    best_state  = copy.deepcopy(model.state_dict())
    best_stats  = {"val_loss": float("inf"), "val_top1_precision": 0.0}
    patience    = 0

    epoch_iter = (
        tqdm(range(cfg.model.max_epochs), desc="Training", unit="epoch")
        if HAS_TQDM else range(cfg.model.max_epochs)
    )

    for _ in epoch_iter:
        model.train()
        train_loss = 0.0
        n_batches  = 0

        for bx, by in loader:
            bx = bx.to(device, non_blocking=pm)
            by = by.to(device, non_blocking=pm)
            optimizer.zero_grad()

            with autocast_fn():
                logits = model(bx)
                loss   = criterion(logits, by)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += float(loss.item())
            n_batches  += 1

        # ── Validation ──────────────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        val_n        = 0
        val_scores: list[float] = []

        with inference_context():
            for bx, by in val_loader:
                bx = bx.to(device, non_blocking=pm)
                by = by.to(device, non_blocking=pm)
                with autocast_fn():
                    logits = model(bx)
                    loss   = criterion(logits, by)
                bs = int(by.shape[0])
                val_loss_sum += float(loss.item()) * bs
                val_n        += bs
                val_scores.extend(
                    torch.sigmoid(logits).float().cpu().numpy().reshape(-1).tolist()
                )

        val_loss = (val_loss_sum / max(1, val_n)) if val_n > 0 else float("inf")
        val_top1 = _daily_topk_precision(val_dates, np.asarray(val_scores), y_val, k=1)

        metric = val_top1 - 0.05 * val_loss
        if metric > best_metric + 1e-8:
            best_metric = metric
            best_state  = copy.deepcopy(model.state_dict())
            best_stats  = {"val_loss": float(val_loss), "val_top1_precision": float(val_top1)}
            patience    = 0
        else:
            patience += 1
            if patience >= int(cfg.model.early_stop_patience):
                break

        if HAS_TQDM:
            epoch_iter.set_postfix(
                train_loss=f"{(train_loss / max(1, n_batches)):.4f}",
                val_loss=f"{val_loss:.4f}",
                val_top1=f"{val_top1:.4f}",
            )

    model.load_state_dict(best_state)
    model.eval()
    return model, best_stats


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def train_walk_forward(cfg: AppConfig, paths: RunPaths, feats: pd.DataFrame, alpha_lib) -> str:
    """Train strategy model across walk-forward folds and persist artifacts."""
    from src.core.logging import get_logger

    log = get_logger(__name__)
    seed_everything(cfg.project.seed)

    if cfg.execution.signal_timing != "EOD_NEXT_OPEN":
        log.warning(
            "Training labels are defined for next-day-open execution. "
            "signal_timing=%s may be misaligned.",
            cfg.execution.signal_timing,
        )

    feature_cols = alpha_lib.feature_names + list(alpha_lib.formulas.keys())
    tickers = paths.load_tickers(cfg.universe.tickers_file)
    df = feats.copy()
    if tickers and "ticker" in df.columns:
        df = df[df["ticker"].isin(set(tickers))].copy()
    if cfg.universe.min_price > 0 and "Close" in df.columns:
        df = df[df["Close"] >= float(cfg.universe.min_price)].copy()
    if cfg.universe.min_median_turnover_20d > 0 and "turnover_med_20d" in df.columns:
        df = df[df["turnover_med_20d"] >= float(cfg.universe.min_median_turnover_20d)].copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    if alpha_lib and alpha_lib.formulas:
        log.info("[1/6] Computing %s alphas...", len(alpha_lib.formulas))
        from src.llm_alpha.alpha_executor import compute_alphas_on_df
        df = compute_alphas_on_df(
            df,
            formulas=alpha_lib.formulas,
            feature_names=alpha_lib.feature_names,
            max_chars=cfg.llm_alpha.max_expr_chars,
            max_window=cfg.llm_alpha.max_rolling_window,
            logger=log,
        )

    log.info("[2/6] Building labels and cleaning...")
    if "label" not in df.columns:
        df = add_label_column_barrier(cfg, df)
    df = df.dropna(subset=["label"] + feature_cols).copy()
    df["date"] = pd.to_datetime(df["date"])
    if df.empty:
        raise ValueError("No data left after alpha/label/dropna.")

    lookback     = int(cfg.features.lookback_days_for_model)
    unique_dates = np.asarray(sorted(df["date"].unique()))
    splits       = _build_walk_forward_splits(unique_dates, cfg)
    if not splits:
        raise ValueError("Not enough history for configured walk-forward windows.")
    log.info("[3/6] Walk-forward folds: %s", len(splits))

    device = get_device(cfg)
    log_device_info(device, log)

    oos_rows:    list[pd.DataFrame]       = []
    fold_rows:   list[dict]               = []
    fold_models: list[tuple[nn.Module, dict]] = []  # (model, split_dict)
    final_bundle: ModelBundle | None      = None

    for i, sp in enumerate(splits, start=1):
        log.info(
            "[4/6] Fold %s/%s | train=%s..%s val=%s..%s test=%s..%s",
            i, len(splits),
            sp["train_start"].date(), sp["train_end"].date(),
            sp["val_start"].date(),   sp["val_end"].date(),
            sp["test_start"].date(),  sp["test_end"].date(),
        )

        train_mask = (df["date"] >= sp["train_start"]) & (df["date"] <= sp["train_end"])
        if train_mask.sum() == 0:
            log.warning("Fold %s skipped: empty train slice", i)
            continue

        scaler = StandardScaler()
        scaler.fit(df.loc[train_mask, feature_cols])
        sdf = df.copy()
        sdf[feature_cols] = scaler.transform(sdf[feature_cols])

        X_train, y_train, _, _           = _build_labeled_windows(sdf, feature_cols, lookback, cfg.execution.signal_timing, sp["train_start"], sp["train_end"])
        X_val,   y_val,   _, val_dates   = _build_labeled_windows(sdf, feature_cols, lookback, cfg.execution.signal_timing, sp["val_start"],   sp["val_end"])
        X_test,  y_test,  test_tickers, test_dates = _build_labeled_windows(sdf, feature_cols, lookback, cfg.execution.signal_timing, sp["test_start"],  sp["test_end"])

        if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
            log.warning(
                "Fold %s skipped: insufficient windows (train=%s, val=%s, test=%s).",
                i, len(X_train), len(X_val), len(X_test),
            )
            continue

        seed_everything(cfg.project.seed + i)
        model, best_stats = _train_fold_model(
            cfg=cfg, device=device,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            val_dates=val_dates,
        )
        # Compile the model for faster inference on CUDA/MPS
        compiled_model = try_compile(model, device) if getattr(cfg.model, "torch_compile", True) else model
        test_scores = _predict_scores(model=compiled_model, X=X_test, device=device)
        test_top1   = _daily_topk_precision(test_dates, test_scores, y_test, k=1)
        test_top2   = _daily_topk_precision(test_dates, test_scores, y_test, k=2)

        fold_rows.append({
            "fold":                     i,
            "train_windows":            int(len(X_train)),
            "val_windows":              int(len(X_val)),
            "test_windows":             int(len(X_test)),
            "val_top1_precision":       float(best_stats["val_top1_precision"]),
            "val_loss":                 float(best_stats["val_loss"]),
            "test_top1_precision":      float(test_top1),
            "test_top2_pick_precision": float(test_top2),
        })

        fold_oos = pd.DataFrame({
            "fold":      i,
            "date":      pd.to_datetime(test_dates),
            "ticker":    test_tickers,
            "label":     y_test.astype(float),
            "raw_score": test_scores.astype(float),
        })
        oos_rows.append(fold_oos)

        model = model.cpu()
        model.eval()
        fold_models.append((model, sp))

    if not fold_models:
        raise RuntimeError("All walk-forward folds were skipped; model was not trained.")

    # Use the last fold's model as the production model, but fit the scaler
    # on ALL training data up to and including the last fold's train period
    # to avoid the scaler mismatch that caused 0-trade backtests.
    last_model, last_split = fold_models[-1]
    cumulative_train_mask = df["date"] <= last_split["train_end"]
    cumulative_scaler = StandardScaler()
    cumulative_scaler.fit(df.loc[cumulative_train_mask, feature_cols])

    final_bundle = ModelBundle(
        model=last_model, scaler=cumulative_scaler,
        feature_cols=feature_cols, lookback=lookback,
        signal_timing=cfg.execution.signal_timing,
    )
    # Attach fold models for potential ensemble use
    final_bundle.fold_models = [(m.state_dict(), sp) for m, sp in fold_models]

    log.info("[5/6] Saving model and OOS calibration artifacts...")
    model_path = paths.model_bundle_path()
    torch.save(final_bundle, model_path)

    folds_df = pd.DataFrame(fold_rows)
    if not folds_df.empty:
        folds_df.to_csv(paths.runs / "reports" / "walk_forward_folds.csv", index=False)

    oos_df = pd.concat(oos_rows, ignore_index=True) if oos_rows else pd.DataFrame()
    if oos_df.empty:
        calibration_obj = {"method": "identity", "top1_threshold": 1.0}
    else:
        cal    = fit_platt_calibrator(oos_df["raw_score"].to_numpy(), oos_df["label"].to_numpy())
        oos_df["calibrated_prob"] = apply_calibration(oos_df["raw_score"].to_numpy(), cal)
        th     = build_thresholds_from_oos(
            oos_df=oos_df[["date", "ticker", "label", "calibrated_prob"]].rename(columns={"calibrated_prob": "score"}),
            score_col="score",
            target_top1_precision=float(cfg.decision.target_top1_precision),
            min_days_for_threshold=int(cfg.decision.min_days_for_threshold),
        )
        calibration_obj = {
            "method":                    cal.get("method", "identity"),
            "coef":                      cal.get("coef"),
            "intercept":                 cal.get("intercept"),
            "top1_threshold":            float(th["top1_threshold"]),
            "top1_threshold_days":       int(th["top1_days"]),
            "top1_threshold_precision":  float(th["top1_precision"]),
            "signal_timing":             cfg.execution.signal_timing,
        }

    if not oos_df.empty:
        oos_df.to_csv(paths.oos_predictions_path(), index=False)
        save_monitoring_reports(cfg, paths, oos_df[["date", "ticker", "label", "calibrated_prob"]])
    paths.model_calibration_path().write_text(json.dumps(calibration_obj, indent=2))

    log.info("[6/6] Training complete. model=%s calibration=%s", model_path, paths.model_calibration_path())
    return str(model_path)

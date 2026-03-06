from __future__ import annotations
"""Walk-forward model training, OOS scoring, calibration, and monitoring."""

import contextlib
import copy
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.backtest.simulator import add_label_column_barrier
from src.core.config import AppConfig
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

# ── FIXED: only force spawn on macOS; Linux (Kaggle) works best with fork ──
if multiprocessing.get_start_method(allow_none=True) is None:
    _default_method = "spawn" if os.uname().sysname == "Darwin" else "fork"
    multiprocessing.set_start_method(_default_method, force=True)
# ───────────────────────────────────────────────────────────────────────────

NUM_PHYSICAL_CORES = multiprocessing.cpu_count()
torch.set_num_threads(NUM_PHYSICAL_CORES)
torch.set_num_interop_threads(NUM_PHYSICAL_CORES)
os.environ["OMP_NUM_THREADS"] = str(NUM_PHYSICAL_CORES)
os.environ["MKL_NUM_THREADS"] = str(NUM_PHYSICAL_CORES)
os.environ["OPENBLAS_NUM_THREADS"] = str(NUM_PHYSICAL_CORES)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(NUM_PHYSICAL_CORES)


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def get_device(cfg: AppConfig) -> torch.device:
    """Resolve training device: CUDA > MPS > CPU."""
    # ── FIXED: default use_cuda_if_available to True so CUDA is auto-used ──
    if getattr(cfg.model, "use_cuda_if_available", True) and torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(cfg.model, "use_mps_if_available", True) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
    # ────────────────────────────────────────────────────────────────────────


def _pin_memory_for(device: torch.device) -> bool:
    """pin_memory is only beneficial (and safe) with CUDA."""
    return device.type == "cuda"


def _make_amp_handles(device: torch.device):
    """
    Return (autocast_ctx_fn, scaler_or_None) for the given device.

    - CUDA  → torch.amp.autocast(float16) + GradScaler
    - MPS   → nullcontext + no scaler  (MPS autocast is still experimental)
    - CPU   → nullcontext + no scaler
    """
    if device.type == "cuda":
        autocast_fn = lambda: torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        scaler = torch.amp.GradScaler("cuda")
    else:
        autocast_fn = contextlib.nullcontext
        scaler = None
    return autocast_fn, scaler


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
# Alpha computation
# ---------------------------------------------------------------------------

def _compute_single_alpha(args):
    """Evaluate one alpha expression on provided feature payload."""
    alpha_name, formula, feats_dict, index, group_col_values = args
    feats = pd.DataFrame(feats_dict, index=index)
    group_col = pd.Series(group_col_values, index=index) if group_col_values is not None else None

    def apply_grouped(s, func):
        if group_col is not None:
            return s.groupby(group_col).transform(func)
        return func(s)

    def _to_series(x, ref=None):
        if isinstance(x, pd.Series):
            return x
        if ref is not None and isinstance(ref, pd.Series):
            return pd.Series(x, index=ref.index)
        return pd.Series(x, index=feats.index)

    def safe_div(x, y):
        x, y = _to_series(x), _to_series(y)
        return x.div(y.replace(0, np.nan)).fillna(0.0)

    def zscore(x, w):
        s = _to_series(x)
        mean = apply_grouped(s, lambda col: col.rolling(int(w)).mean())
        std  = apply_grouped(s, lambda col: col.rolling(int(w)).std())
        return (s - mean).div(std.replace(0, np.nan)).fillna(0.0)

    def delta(x, p):
        return apply_grouped(_to_series(x), lambda col: col.diff(int(p)))

    def rolling_mean(x, w):
        return apply_grouped(_to_series(x), lambda col: col.rolling(int(w)).mean())

    def rolling_std(x, w):
        return apply_grouped(_to_series(x), lambda col: col.rolling(int(w)).std()).fillna(0.0)

    def rolling_min(x, w):
        return apply_grouped(_to_series(x), lambda col: col.rolling(int(w)).min())

    def rolling_max(x, w):
        return apply_grouped(_to_series(x), lambda col: col.rolling(int(w)).max())

    def ewm_mean(x, span):
        return apply_grouped(_to_series(x), lambda col: col.ewm(span=int(span)).mean())

    def sign(x):
        s = _to_series(x)
        return pd.Series(np.sign(s.values), index=s.index)

    def clip(x, lo, hi):
        return _to_series(x).clip(lower=lo, upper=hi)

    def log1p(x):
        s = _to_series(x)
        return pd.Series(np.log1p(s.values), index=s.index)

    def shift(x, d):
        return apply_grouped(_to_series(x), lambda col: col.shift(int(d)))

    eval_globals = {
        "safe_div": safe_div, "zscore": zscore, "delta": delta,
        "rolling_mean": rolling_mean, "rolling_std": rolling_std,
        "rolling_min": rolling_min, "rolling_max": rolling_max,
        "sign": sign, "abs": lambda x: _to_series(x).abs(),
        "np": np, "clip": clip, "log1p": log1p,
        "shift": shift, "ewm_mean": ewm_mean,
    }
    try:
        result = eval(formula, eval_globals, {col: feats[col] for col in feats.columns})
        vals = result.values if isinstance(result, pd.Series) else np.array(result)
        return alpha_name, vals, None
    except Exception as e:
        return alpha_name, None, str(e)


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
    pin_memory: bool = False,       # ← CUDA: True only when device is CUDA
) -> DataLoader:
    """Wrap NumPy arrays into a torch DataLoader."""
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y.reshape(-1, 1), dtype=torch.float32),
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin_memory,
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _predict_scores(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    """Run batched inference and return sigmoid scores."""
    if len(X) == 0:
        return np.empty((0,), dtype=np.float32)

    pin_memory = _pin_memory_for(device)
    # ── FIXED: was incorrectly calling _make_amp_context (does not exist) ──
    autocast_fn, _ = _make_amp_handles(device)
    # ────────────────────────────────────────────────────────────────────────

    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    out: list[float] = []
    model.eval()
    with torch.no_grad():
        for (bx,) in loader:
            bx = bx.to(device, non_blocking=pin_memory)
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

    pin_memory   = _pin_memory_for(device)
    autocast_fn, scaler = _make_amp_handles(device)

    loader     = _to_loader(X_train, y_train, cfg.model.batch_size, shuffle=True,  pin_memory=pin_memory)
    val_loader = _to_loader(X_val,   y_val,   cfg.model.batch_size, shuffle=False, pin_memory=pin_memory)

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
            bx = bx.to(device, non_blocking=pin_memory)
            by = by.to(device, non_blocking=pin_memory)
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

        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device, non_blocking=pin_memory)
                by = by.to(device, non_blocking=pin_memory)
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
        group_col_values = df["ticker"].tolist() if "ticker" in df.columns else None
        needed_cols  = [c for c in df.columns if c not in alpha_lib.formulas]
        feats_dict   = {col: df[col].tolist() for col in needed_cols}
        tasks = [
            (name, formula, feats_dict, df.index, group_col_values)
            for name, formula in alpha_lib.formulas.items()
            if name not in df.columns
        ]
        n_workers = min(max(1, len(tasks)), NUM_PHYSICAL_CORES)
        completed = 0
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_compute_single_alpha, t): t[0] for t in tasks}
                for fut in as_completed(futures):
                    name, values, err = fut.result()
                    completed += 1
                    if err:
                        log.warning("alpha %s failed (%s); filling 0.0", name, err)
                        df[name] = 0.0
                    else:
                        df[name] = values
                    if completed % 5 == 0 or completed == len(tasks):
                        log.info("alphas done: %s/%s", completed, len(tasks))
        except Exception as e:
            log.warning("Parallel alpha computation unavailable (%s). Falling back to sequential.", e)
            for task in tasks:
                name, values, err = _compute_single_alpha(task)
                df[name] = 0.0 if err else values
                if err:
                    log.warning("alpha %s failed (%s); filling 0.0", name, err)

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
    log.info("Training device: %s", device)
    if device.type == "cuda":
        log.info("  CUDA device: %s | VRAM: %.1f GB",
                 torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1e9)

    oos_rows:    list[pd.DataFrame]       = []
    fold_rows:   list[dict]               = []
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
        test_scores = _predict_scores(model=model, X=X_test, device=device, batch_size=1024)
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
        final_bundle = ModelBundle(
            model=model, scaler=scaler,
            feature_cols=feature_cols, lookback=lookback,
            signal_timing=cfg.execution.signal_timing,
        )

    if final_bundle is None:
        raise RuntimeError("All walk-forward folds were skipped; model was not trained.")

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

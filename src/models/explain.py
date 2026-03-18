from __future__ import annotations

"""Feature importance and explainability helpers."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


BASE_FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "returns_1d", "returns_5d", "RSI_14", "MACD", "EMA_20", "EMA_50", "ATR_14", "vol_z",
    "turnover", "turnover_med_20d",
]


def feature_schema_from_df(df: pd.DataFrame) -> list[str]:
    """Return allowed base feature names present in dataframe."""
    cols = [c for c in BASE_FEATURES if c in df.columns]
    return cols


def permutation_feature_importance(
    model_bundle,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute permutation-based feature importance for the model.

    For each feature, shuffles it across all timesteps in the window
    and measures the drop in mean prediction accuracy.

    Args:
        model_bundle: ModelBundle with .model and .scaler
        X: (N, seq_len, n_features) array of windows.
        y: (N,) binary labels.
        feature_names: Names corresponding to X's last dimension.
        n_repeats: Number of shuffle repeats per feature.
        seed: Random seed.

    Returns:
        DataFrame with columns [feature, importance_mean, importance_std]
        sorted by importance descending.
    """
    rng = np.random.default_rng(seed)
    model = model_bundle.model
    device = next(model.parameters()).device
    model.eval()

    def _score(X_in: np.ndarray) -> float:
        """Compute top-1 daily precision proxy (mean label of top predictions)."""
        t = torch.tensor(X_in, dtype=torch.float32)
        ds = TensorDataset(t)
        loader = DataLoader(ds, batch_size=512, shuffle=False)
        preds = []
        with torch.no_grad():
            for (bx,) in loader:
                bx = bx.to(device)
                logits = model(bx)
                probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
                preds.extend(probs.tolist())
        preds = np.array(preds)
        # Use AUC-like metric: correlation between prediction and label
        if np.std(preds) < 1e-10:
            return 0.0
        return float(np.corrcoef(preds, y)[0, 1])

    baseline_score = _score(X)
    results = []

    for feat_idx, feat_name in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            # Shuffle this feature across samples (all timesteps)
            perm_idx = rng.permutation(len(X_perm))
            X_perm[:, :, feat_idx] = X_perm[perm_idx, :, feat_idx]
            perm_score = _score(X_perm)
            drops.append(baseline_score - perm_score)
        results.append({
            "feature": feat_name,
            "importance_mean": float(np.mean(drops)),
            "importance_std": float(np.std(drops)),
        })

    return pd.DataFrame(results).sort_values("importance_mean", ascending=False).reset_index(drop=True)

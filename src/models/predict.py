"""Batch prediction helper for window datasets."""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def predict_proba(model_bundle, X: np.ndarray, use_mps: bool = True) -> np.ndarray:
    """Predict probabilities for a prebuilt window dataset object.

    Note:
        This helper expects `X` to behave like the project window dataset
        (it reuses `dataset.data_arr`). The main production path currently uses
        direct inference in training/backtest/signal modules.
    """
    model = model_bundle.model
    scaler = model_bundle.scaler

    # 1. Scale the data (flatten, scale, reshape back)
    B, seq_len, num_feats = X.shape
    X_flat = X.reshape(-1, num_feats)
    X_scaled = scaler.transform(X_flat)
    X_scaled = X_scaled.reshape(B, seq_len, num_feats)

    # 2. Setup Device
    # ── FIXED: prefer CUDA (Kaggle/cloud), fall back to MPS (Mac), then CPU ──
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif use_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # ─────────────────────────────────────────────────────────────────────────

    model.to(device)
    model.eval()

    # 3. Create a DataLoader to batch the predictions and avoid OOM
    dataset = X

    # Re-inject the scaled data back into the dataset before predicting
    dataset.data_arr = X_scaled

    # Larger batch size for CUDA; keep small for MPS to prevent memory spikes
    batch_size = 512 if device.type == "cuda" else 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_probs = []

    # 4. Run inference in batches
    with torch.no_grad():
        for batch in dataloader:
            xb = batch[0].to(device)
            logit = model(xb).detach().cpu().numpy()

            # Convert logits to probabilities using sigmoid
            prob = 1.0 / (1.0 + np.exp(-logit.squeeze(-1)))

            # If the output is a scalar (batch size 1), make it a list
            if prob.ndim == 0:
                all_probs.append(prob.item())
            else:
                all_probs.extend(prob.tolist())

    return np.array(all_probs)

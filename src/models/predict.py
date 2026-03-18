"""Batch prediction helper for window datasets."""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.core.device import (
    inference_context,
    make_amp_context,
    optimal_batch_size,
    optimal_num_workers,
    pin_memory_for,
    resolve_device,
    tune_for_device,
)


def predict_proba(model_bundle, X: np.ndarray, use_mps: bool = True) -> np.ndarray:
    """Predict probabilities for a prebuilt numpy window array.

    Args:
        model_bundle: ModelBundle with .model and .scaler attributes.
        X: numpy array of shape (B, seq_len, num_feats).
        use_mps: Whether to use MPS (Apple Silicon) if available.

    Returns:
        np.ndarray of predicted probabilities, shape (B,).
    """
    model = model_bundle.model
    scaler = model_bundle.scaler

    # 1. Scale the data (flatten, scale, reshape back)
    B, seq_len, num_feats = X.shape
    X_flat = X.reshape(-1, num_feats)
    X_scaled = scaler.transform(X_flat)
    X_scaled = X_scaled.reshape(B, seq_len, num_feats)

    # 2. Setup Device
    device = resolve_device(prefer_cuda=True, prefer_mps=use_mps)
    tune_for_device(device)

    model.to(device)
    model.eval()

    # 3. Create a DataLoader from the scaled numpy array
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)

    bs = optimal_batch_size(device)["inference"]
    pm = pin_memory_for(device)
    nw = optimal_num_workers(device)
    autocast_fn, _ = make_amp_context(device)

    dataloader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=pm,
        persistent_workers=(nw > 0),
    )

    all_probs = []

    # 4. Run inference in batches with AMP + inference_mode
    with inference_context():
        for (batch,) in dataloader:
            xb = batch.to(device, non_blocking=pm)
            with autocast_fn():
                logit = model(xb)
            prob = torch.sigmoid(logit).float().cpu().numpy().squeeze(-1)

            if prob.ndim == 0:
                all_probs.append(prob.item())
            else:
                all_probs.extend(prob.tolist())

    return np.array(all_probs)

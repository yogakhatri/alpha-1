"""Random-seed utilities for reproducible model training runs."""

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs (including MPS when available)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)  # type: ignore[attr-defined]

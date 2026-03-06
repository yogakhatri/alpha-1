"""Random-seed utilities for reproducible model training runs."""

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs (CUDA, MPS, or CPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)  # type: ignore[attr-defined]

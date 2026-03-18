"""Random-seed utilities for reproducible model training runs."""

import os
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
        # Deterministic ops at the cost of a small perf hit;
        # comment out if speed matters more than exact reproducibility.
        torch.backends.cudnn.deterministic = False  # False = faster (benchmark mode)
    # MPS seed is only available in PyTorch ≥ 2.1
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except (AttributeError, RuntimeError):
            pass
    # Ensure hash-based operations are deterministic across workers
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

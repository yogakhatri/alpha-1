from __future__ import annotations
"""Dataset wrappers for building fixed-lookback model windows."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TimeSeriesWindowDataset(Dataset):
    """
    Memory-efficient PyTorch Dataset that generates sliding windows on-the-fly.
    Windows never cross ticker boundaries. Zero extra RAM beyond the DataFrame.
    """
    def __init__(self, df: pd.DataFrame, feature_cols: list[str], lookback: int, label_col: str = None):
        self.lookback = lookback
        self.feature_cols = feature_cols
        self.label_col = label_col

        # ── FIX: Sort by ticker first to guarantee contiguous ticker blocks ──
        if 'ticker' in df.columns:
            df = df.sort_values('ticker', kind='stable').reset_index(drop=True)

        # Store raw data as contiguous NumPy arrays (fast random access)
        self.data_arr  = df[feature_cols].to_numpy(dtype=np.float32)
        self.label_arr = df[label_col].to_numpy(dtype=np.float32) if (label_col and label_col in df.columns) else None

        # ── FIX: Use groupby to find exact row positions per ticker ──────────
        self.valid_indices = []

        if 'ticker' in df.columns:
            # cumcount gives the position of each row within its ticker group
            df['_pos'] = np.arange(len(df))
            for ticker, grp in df.groupby('ticker', sort=False):
                positions = grp['_pos'].values   # absolute row positions in data_arr
                count     = len(positions)
                if count < lookback:
                    continue
                # valid window end positions: from lookback-1 onward within this ticker
                for i in range(lookback - 1, count):
                    self.valid_indices.append(positions[i])
            df.drop(columns=['_pos'], inplace=True)
        else:
            # No ticker column — treat entire DataFrame as one sequence
            N = len(df)
            self.valid_indices = list(range(lookback - 1, N))

        self.valid_indices = np.array(self.valid_indices, dtype=np.int32)

    def __len__(self):
        """Number of valid end-indices/windows in dataset."""
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """Return one feature window and optional binary label."""
        end_idx   = int(self.valid_indices[idx])
        start_idx = end_idx - self.lookback + 1

        # torch.from_numpy is faster than torch.tensor (zero-copy)
        X_window = torch.from_numpy(self.data_arr[start_idx:end_idx + 1].copy())

        if self.label_arr is not None:
            y_val = torch.tensor([self.label_arr[end_idx]], dtype=torch.float32)
            return X_window, y_val

        return X_window


def make_windows(df: pd.DataFrame, feature_cols: list[str], lookback: int, label_col: str = None):
    """Construct lazy window dataset from dataframe."""
    dataset = TimeSeriesWindowDataset(df, feature_cols, lookback, label_col)
    return dataset, None

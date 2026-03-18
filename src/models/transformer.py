"""Transformer classifier for fixed-length financial time windows."""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal ordering in sequences."""
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):
    """Sequence encoder + classification head for binary swing target."""
    def __init__(self, num_features, d_model=64, n_heads=4, n_layers=3, dropout=0.2, num_classes=1):
        super().__init__()
        # Project raw features into the d_model dimension
        self.feature_proj = nn.Linear(num_features, d_model)

        # Positional encoding for temporal ordering
        self.pos_enc = PositionalEncoding(d_model, max_len=512, dropout=dropout)
        
        # Transformer Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model * 4, 
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        
        # Final classification head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        """Forward pass returning logits for BCEWithLogitsLoss."""
        # x shape: (Batch, Sequence Length, Features)
        x = self.feature_proj(x)

        # Add positional encoding
        x = self.pos_enc(x)
        
        # Pass through transformer
        x = self.encoder(x)
        
        # Pool the sequence (take the features from the last time step)
        # x[:, -1, :] shape: (Batch, d_model)
        last_step = x[:, -1, :]
        
        # Output logits
        out = self.head(last_step)
        return out

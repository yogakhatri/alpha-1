"""Transformer classifier for fixed-length financial time windows."""

import torch
import torch.nn as nn

class TimeSeriesTransformer(nn.Module):
    """Sequence encoder + classification head for binary swing target."""
    def __init__(self, num_features, d_model=64, n_heads=4, n_layers=3, dropout=0.2, num_classes=1):
        super().__init__()
        # Project raw features into the d_model dimension
        self.feature_proj = nn.Linear(num_features, d_model)
        
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
        
        # Pass through transformer
        x = self.encoder(x)
        
        # Pool the sequence (take the features from the last time step)
        # x[:, -1, :] shape: (Batch, d_model)
        last_step = x[:, -1, :]
        
        # Output logits
        out = self.head(last_step)
        return out

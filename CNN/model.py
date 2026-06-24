from __future__ import annotations

import torch
from torch import nn


class IndustryCNN(nn.Module):
    """CNN baseline over the industry-feature matrix.

    Input shape is ``(batch, num_nodes, num_features)``. The model treats each
    daily cross section as a small image with one channel, where rows are
    industries and columns are node features. It outputs one future excess
    return prediction for each industry.
    """

    def __init__(
        self,
        num_features: int,
        hidden_channels: int = 32,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
        )
        self.feature_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.regressor = nn.Linear(hidden_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected x with shape (batch, nodes, features), got {tuple(x.shape)}")

        h = self.encoder(x.unsqueeze(1))
        h = self.feature_pool(h).squeeze(-1)
        h = h.transpose(1, 2)
        return self.regressor(h).squeeze(-1)

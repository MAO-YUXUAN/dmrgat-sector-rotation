from __future__ import annotations

import torch
from torch import nn


class IndustryLSTM(nn.Module):
    """LSTM baseline over each industry's historical feature sequence.

    Input shape is ``(batch, seq_len, num_nodes, num_features)``. The model
    applies the same LSTM to every industry node and predicts one future excess
    return for each industry at the final date of the input sequence.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                "Expected x with shape (batch, seq_len, nodes, features), "
                f"got {tuple(x.shape)}"
            )

        batch_size, seq_len, num_nodes, num_features = x.shape
        node_sequences = x.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, seq_len, num_features)
        output, _ = self.lstm(node_sequences)
        final_state = self.dropout(output[:, -1, :])
        preds = self.regressor(final_state).reshape(batch_size, num_nodes)
        return preds

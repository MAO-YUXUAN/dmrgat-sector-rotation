from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class WeightedGATLayer(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        concat: bool = True,
    ) -> None:
        super().__init__(aggr="add", node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.concat = concat

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att = nn.Parameter(torch.empty(heads, 2 * out_channels))
        self.bias = nn.Parameter(torch.zeros(heads * out_channels if concat else out_channels))
        self._alpha: torch.Tensor | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        h = self.lin(x).view(-1, self.heads, self.out_channels)
        out = self.propagate(edge_index=edge_index, x=h, edge_weight=edge_weight)
        alpha = self._alpha
        self._alpha = None
        if self.concat:
            out = out.reshape(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        out = out + self.bias
        if return_attention_weights:
            if alpha is None:
                raise RuntimeError("Attention weights were not computed.")
            return out, (edge_index, alpha)
        return out

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_weight: torch.Tensor,
        index: torch.Tensor,
        ptr: torch.Tensor | None,
        size_i: int | None,
    ) -> torch.Tensor:
        pair = torch.cat([x_i, x_j], dim=-1)
        score = (pair * self.att.unsqueeze(0)).sum(dim=-1)
        score = F.leaky_relu(score, negative_slope=self.negative_slope)
        score = score * edge_weight.view(-1, 1)
        alpha = softmax(score, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        self._alpha = alpha
        return x_j * alpha.unsqueeze(-1)


class DMRGAT(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        out_dim: int = 1,
    ) -> None:
        super().__init__()
        self.layer1 = WeightedGATLayer(in_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=True)
        self.layer2 = WeightedGATLayer(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout, concat=False)
        self.norm1 = nn.LayerNorm(hidden_dim * num_heads)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x, edge_index, edge_weight)
        x = self.norm1(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.layer2(x, edge_index, edge_weight)
        x = self.norm2(x)
        x = F.elu(x)
        x = self.dropout(x)
        return self.regressor(x).squeeze(-1)

    def get_attention_maps(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        x1, attn1 = self.layer1(x, edge_index, edge_weight, return_attention_weights=True)
        x1 = self.norm1(x1)
        x1 = F.elu(x1)
        x1 = self.dropout(x1)
        _, attn2 = self.layer2(x1, edge_index, edge_weight, return_attention_weights=True)
        return {
            "layer1": attn1,
            "layer2": attn2,
        }

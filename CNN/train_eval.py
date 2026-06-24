from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, Dataset

from dmrgat.config import ProjectConfig
from dmrgat.graph_builder import GraphDataset, split_graphs
from dmrgat.utils import get_device, set_seed

from model import IndustryCNN


@dataclass
class CNNRunResult:
    train_rmse: float
    val_rmse: float
    test_rmse: float
    train_mae: float
    val_mae: float
    test_mae: float
    train_ic: float
    val_ic: float
    test_ic: float
    train_rank_ic: float
    val_rank_ic: float
    test_rank_ic: float
    num_graphs: int
    num_nodes: int
    num_features: int


class GraphTensorDataset(Dataset):
    def __init__(self, graphs) -> None:
        self.graphs = graphs

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        graph = self.graphs[index]
        return graph.x.float(), graph.y.float()


def _pairwise_ranking_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    pred_diff = preds.unsqueeze(-1) - preds.unsqueeze(-2)
    target_diff = targets.unsqueeze(-1) - targets.unsqueeze(-2)
    sign = torch.sign(target_diff)
    valid_mask = sign != 0
    if not torch.any(valid_mask):
        return preds.new_tensor(0.0)

    signed_margin = pred_diff * sign
    losses = F.softplus(-signed_margin)
    return losses[valid_mask].mean()


def _corr_or_nan(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    if np.allclose(np.std(a), 0.0) or np.allclose(np.std(b), 0.0):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _nanmean_or_nan(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.isnan(arr).all():
        return float("nan")
    return float(np.nanmean(arr))


def _make_loader(graphs, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(GraphTensorDataset(graphs), batch_size=batch_size, shuffle=shuffle)


def _run_epoch(
    model: IndustryCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    ranking_loss_weight: float,
    mse_loss_weight: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        preds = model(x)
        rank_loss = _pairwise_ranking_loss(preds, y)
        mse_loss = F.mse_loss(preds, y)
        loss = ranking_loss_weight * rank_loss + mse_loss_weight * mse_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_batches += 1
    return total_loss / max(total_batches, 1)


@torch.no_grad()
def _evaluate(model: IndustryCNN, graphs, device: torch.device, batch_size: int) -> dict[str, float]:
    model.eval()
    if not graphs:
        return {key: float("nan") for key in ["rmse", "mae", "ic", "rank_ic"]}

    preds_all = []
    labels_all = []
    daily_ic = []
    daily_rank_ic = []

    loader = _make_loader(graphs, batch_size=batch_size, shuffle=False)
    for x, y in loader:
        x = x.to(device)
        batch_pred = model(x).cpu().numpy()
        batch_true = y.numpy()
        preds_all.append(batch_pred.reshape(-1))
        labels_all.append(batch_true.reshape(-1))

        for pred_row, true_row in zip(batch_pred, batch_true):
            daily_ic.append(_corr_or_nan(pred_row, true_row))
            daily_rank_ic.append(
                _corr_or_nan(
                    pd.Series(pred_row).rank().to_numpy(),
                    pd.Series(true_row).rank().to_numpy(),
                )
            )

    y_pred = np.concatenate(preds_all)
    y_true = np.concatenate(labels_all)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "ic": _nanmean_or_nan(daily_ic),
        "rank_ic": _nanmean_or_nan(daily_rank_ic),
    }


def fit_and_evaluate_cnn(
    graph_dataset: GraphDataset,
    config: ProjectConfig,
    batch_size: int = 32,
    hidden_channels: int | None = None,
) -> tuple[CNNRunResult, IndustryCNN]:
    set_seed(config.seed)
    splits = split_graphs(graph_dataset, config)
    device = get_device(config.use_gpu)
    hidden_channels = hidden_channels or config.hidden_dim

    model = IndustryCNN(
        num_features=len(graph_dataset.feature_names),
        hidden_channels=hidden_channels,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    train_loader = _make_loader(splits["train"], batch_size=batch_size, shuffle=True)
    best_state = None
    best_val = float("-inf")
    patience_left = config.patience

    for _ in range(config.epochs):
        _run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            ranking_loss_weight=config.ranking_loss_weight,
            mse_loss_weight=config.mse_loss_weight,
        )
        val_metrics = _evaluate(model, splits["val"], device, batch_size=batch_size)
        val_score = val_metrics["rank_ic"]
        if np.isnan(val_score):
            val_score = -val_metrics["rmse"]
        if val_score > best_val:
            best_val = val_score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = config.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_metrics = _evaluate(model, splits["train"], device, batch_size=batch_size)
    val_metrics = _evaluate(model, splits["val"], device, batch_size=batch_size)
    test_metrics = _evaluate(model, splits["test"], device, batch_size=batch_size)

    result = CNNRunResult(
        train_rmse=train_metrics["rmse"],
        val_rmse=val_metrics["rmse"],
        test_rmse=test_metrics["rmse"],
        train_mae=train_metrics["mae"],
        val_mae=val_metrics["mae"],
        test_mae=test_metrics["mae"],
        train_ic=train_metrics["ic"],
        val_ic=val_metrics["ic"],
        test_ic=test_metrics["ic"],
        train_rank_ic=train_metrics["rank_ic"],
        val_rank_ic=val_metrics["rank_ic"],
        test_rank_ic=test_metrics["rank_ic"],
        num_graphs=len(graph_dataset.graphs),
        num_nodes=len(graph_dataset.node_names),
        num_features=len(graph_dataset.feature_names),
    )
    return result, model


def save_metrics(metrics: CNNRunResult, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(metrics), indent=2, ensure_ascii=False), encoding="utf-8")

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import ProjectConfig
from .graph_builder import GraphDataset, split_graphs
from .model import DMRGAT
from .utils import get_device, set_seed


@dataclass
class RunResult:
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


def _pairwise_ranking_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    pred_diff = preds.unsqueeze(1) - preds.unsqueeze(0)
    target_diff = targets.unsqueeze(1) - targets.unsqueeze(0)
    sign = torch.sign(target_diff)
    valid_mask = sign != 0
    if not torch.any(valid_mask):
        return preds.new_tensor(0.0)

    signed_margin = pred_diff * sign
    losses = F.softplus(-signed_margin)
    return losses[valid_mask].mean()


def _run_epoch(model: DMRGAT, graphs, optimizer, device: torch.device) -> float:
    if not graphs:
        return 0.0
    model.train()
    total_loss = 0.0
    for graph in graphs:
        graph = graph.to(device)
        optimizer.zero_grad()
        preds = model(graph.x, graph.edge_index, graph.edge_attr)
        rank_loss = _pairwise_ranking_loss(preds, graph.y)
        mse_loss = F.mse_loss(preds, graph.y)
        loss = (
            rank_loss * optimizer.param_groups[0].get("ranking_loss_weight", 1.0)
            + mse_loss * optimizer.param_groups[0].get("mse_loss_weight", 0.2)
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(graphs), 1)


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


@torch.no_grad()
def _evaluate(model: DMRGAT, graphs, device: torch.device) -> dict[str, float]:
    model.eval()
    if not graphs:
        return {key: float("nan") for key in ["rmse", "mae", "ic", "rank_ic"]}

    preds = []
    labels = []
    daily_ic = []
    daily_rank_ic = []

    for graph in graphs:
        graph = graph.to(device)
        graph_pred = model(graph.x, graph.edge_index, graph.edge_attr).cpu().numpy()
        graph_true = graph.y.cpu().numpy()
        preds.append(graph_pred)
        labels.append(graph_true)
        daily_ic.append(_corr_or_nan(graph_pred, graph_true))
        daily_rank_ic.append(
            _corr_or_nan(
                pd.Series(graph_pred).rank().to_numpy(),
                pd.Series(graph_true).rank().to_numpy(),
            )
        )

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(labels)

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "ic": _nanmean_or_nan(daily_ic),
        "rank_ic": _nanmean_or_nan(daily_rank_ic),
    }


def fit_and_evaluate(graph_dataset: GraphDataset, config: ProjectConfig) -> RunResult:
    metrics, _ = fit_and_evaluate_with_model(graph_dataset, config)
    return metrics


def fit_and_evaluate_with_model(graph_dataset: GraphDataset, config: ProjectConfig) -> tuple[RunResult, DMRGAT]:
    set_seed(config.seed)
    splits = split_graphs(graph_dataset, config)
    device = get_device(config.use_gpu)

    model = DMRGAT(
        in_dim=len(graph_dataset.feature_names),
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        dropout=config.dropout,
        out_dim=1,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    for group in optimizer.param_groups:
        group["ranking_loss_weight"] = config.ranking_loss_weight
        group["mse_loss_weight"] = config.mse_loss_weight

    best_state = None
    best_val = float("-inf")
    patience_left = config.patience

    for _ in range(config.epochs):
        _run_epoch(model, splits["train"], optimizer, device)
        val_metrics = _evaluate(model, splits["val"], device)
        val_score = val_metrics["rank_ic"]
        if np.isnan(val_score):
            val_score = -val_metrics["rmse"]
        if val_score > best_val:
            best_val = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = config.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_metrics = _evaluate(model, splits["train"], device)
    val_metrics = _evaluate(model, splits["val"], device)
    test_metrics = _evaluate(model, splits["test"], device)

    return (
        RunResult(
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
        ),
        model,
    )

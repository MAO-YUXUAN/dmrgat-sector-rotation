from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from .config import ProjectConfig
from .preprocess import PreparedDataset


@dataclass
class GraphDataset:
    graphs: list[Data]
    dates: list[pd.Timestamp]
    feature_names: list[str]
    node_names: list[str]


def _build_industry_relation_matrix(industry_names: list[str]) -> np.ndarray:
    size = len(industry_names)
    relation = np.eye(size, dtype=float)
    groups = [
        [
            "\u77f3\u6cb9\u77f3\u5316",
            "\u7164\u70ad",
            "\u6709\u8272\u91d1\u5c5e",
            "\u94a2\u94c1",
            "\u57fa\u7840\u5316\u5de5",
        ],
        [
            "\u673a\u68b0",
            "\u7535\u529b\u8bbe\u5907",
            "\u56fd\u9632\u519b\u5de5",
            "\u6c7d\u8f66",
            "\u5bb6\u7535",
        ],
        [
            "\u98df\u54c1\u996e\u6599",
            "\u519c\u6797\u7267\u6e14",
            "\u7eba\u7ec7\u670d\u88c5",
            "\u793e\u4f1a\u670d\u52a1",
            "\u7f8e\u5bb9\u62a4\u7406",
            "\u5546\u8d38\u96f6\u552e",
            "\u8f7b\u5de5\u5236\u9020",
            "\u533b\u836f",
        ],
        [
            "\u7535\u5b50",
            "\u8ba1\u7b97\u673a",
            "\u901a\u4fe1",
            "\u4f20\u5a92",
        ],
        [
            "\u94f6\u884c",
            "\u975e\u94f6\u884c\u91d1\u878d",
            "\u623f\u5730\u4ea7",
            "\u5efa\u7b51",
        ],
        [
            "\u516c\u7528\u4e8b\u4e1a",
            "\u4ea4\u901a\u8fd0\u8f93",
            "\u7efc\u5408",
            "\u5efa\u6750",
            "\u73af\u4fdd",
        ],
    ]

    name_to_idx = {name: idx for idx, name in enumerate(industry_names)}
    for members in groups:
        matched = [name_to_idx[name] for name in industry_names if any(key in name for key in members)]
        for i in matched:
            for j in matched:
                relation[i, j] = 1.0
    return relation


def _pivot_feature(feature_frame: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return feature_frame.pivot(index="trade_date", columns="ts_code", values=value_col).sort_index()


def _corr_to_weight(corr_matrix: pd.DataFrame) -> np.ndarray:
    corr = corr_matrix.fillna(0.0).clip(-1.0, 1.0).to_numpy(dtype=float)
    weights = (corr + 1.0) / 2.0
    np.fill_diagonal(weights, 1.0)
    return weights


def _top_k_mask(matrix: np.ndarray, top_k: int) -> np.ndarray:
    masked = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        row = matrix[i].copy()
        top_idx = np.argsort(row)[-(top_k + 1) :]
        masked[i, top_idx] = row[top_idx]
    masked = np.maximum(masked, masked.T)
    np.fill_diagonal(masked, 1.0)
    return masked


def _dense_to_edge_index(matrix: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    src, dst = np.nonzero(matrix > 0)
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    edge_weight = torch.tensor(matrix[src, dst], dtype=torch.float32)
    return edge_index, edge_weight


def build_dynamic_graphs(dataset: PreparedDataset, config: ProjectConfig) -> GraphDataset:
    frame = dataset.feature_frame.copy()
    industries = dataset.industries.copy()

    code_col = "index_code" if "index_code" in industries.columns else "ts_code"
    name_candidates = ["industry_name", "industry_name1", "name", "industry"]
    name_col = next((col for col in name_candidates if col in industries.columns), None)
    if name_col not in industries.columns:
        fallback_names = frame["ts_code"].drop_duplicates().tolist()
        industries = pd.DataFrame({code_col: fallback_names, "industry_name": fallback_names})
        name_col = "industry_name"

    available_codes = sorted(frame["ts_code"].unique().tolist())
    industries = industries[industries[code_col].isin(available_codes)].drop_duplicates(code_col)
    industries = industries.set_index(code_col).reindex(available_codes).reset_index()
    if name_col not in industries.columns:
        industries[name_col] = industries[code_col]

    node_names = industries[name_col].fillna(industries[code_col]).tolist()
    industry_matrix = _build_industry_relation_matrix(node_names)

    feature_columns = [col for col in frame.columns if col not in {"ts_code", "trade_date", "close", "amount", "target"}]
    returns_pivot = _pivot_feature(frame, "ret_1d")
    fund_pivot = _pivot_feature(frame, "amount_chg")
    dates = sorted(frame["trade_date"].unique().tolist())

    graphs: list[Data] = []
    graph_dates: list[pd.Timestamp] = []

    for current_date in dates:
        return_window = returns_pivot.loc[:current_date].tail(config.corr_window)
        fund_window = fund_pivot.loc[:current_date].tail(config.corr_window)
        if len(return_window) < config.corr_window or len(fund_window) < config.corr_window:
            continue

        day_frame = frame[frame["trade_date"] == current_date].set_index("ts_code").reindex(available_codes)
        if day_frame[feature_columns + ["target"]].isna().any().any():
            continue

        corr_weight = _corr_to_weight(return_window.corr())
        fund_weight = _corr_to_weight(fund_window.corr())
        fused = config.alpha * corr_weight + config.beta * industry_matrix + config.gamma * fund_weight
        fused = fused / max(config.alpha + config.beta + config.gamma, 1e-8)
        fused = _top_k_mask(fused, config.top_k_neighbors)

        edge_index, edge_weight = _dense_to_edge_index(fused)
        x = torch.tensor(day_frame[feature_columns].to_numpy(dtype=np.float32), dtype=torch.float32)
        y = torch.tensor(day_frame["target"].to_numpy(dtype=np.float32), dtype=torch.float32)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)
        data.trade_date = pd.Timestamp(current_date).strftime("%Y-%m-%d")
        graphs.append(data)
        graph_dates.append(pd.Timestamp(current_date))

    return GraphDataset(graphs=graphs, dates=graph_dates, feature_names=feature_columns, node_names=node_names)


def split_graphs(graph_dataset: GraphDataset, config: ProjectConfig) -> dict[str, list[Data]]:
    total = len(graph_dataset.graphs)
    if total < 3:
        raise ValueError("Need at least 3 graph snapshots to form train/val/test splits.")

    train_end = max(int(total * config.train_ratio), 1)
    val_end = max(int(total * (config.train_ratio + config.val_ratio)), train_end + 1)
    val_end = min(val_end, total - 1)
    return {
        "train": graph_dataset.graphs[:train_end],
        "val": graph_dataset.graphs[train_end:val_end],
        "test": graph_dataset.graphs[val_end:],
    }

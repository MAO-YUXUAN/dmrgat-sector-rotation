from __future__ import annotations

import json
from pathlib import Path

from .config import ProjectConfig
from .data_download import download_all
from .graph_builder import build_dynamic_graphs
from .preprocess import build_feature_frame, save_feature_frame
from .train_eval import fit_and_evaluate_with_model
from .utils import ensure_dir
from .visualization import save_attention_heatmap


def run_download(config: ProjectConfig) -> None:
    paths = download_all(config)
    print("Downloaded files:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


def run_train(config: ProjectConfig) -> None:
    dataset = build_feature_frame(config)
    save_feature_frame(dataset, config)
    graph_dataset = build_dynamic_graphs(dataset, config)
    if not graph_dataset.graphs:
        raise RuntimeError("No valid graph snapshots were built. Check date range and raw data completeness.")

    metrics, model = fit_and_evaluate_with_model(graph_dataset, config)
    ensure_dir(config.processed_dir)
    metrics_path = Path(config.processed_dir) / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "train_rmse": metrics.train_rmse,
                "val_rmse": metrics.val_rmse,
                "test_rmse": metrics.test_rmse,
                "train_mae": metrics.train_mae,
                "val_mae": metrics.val_mae,
                "test_mae": metrics.test_mae,
                "train_ic": metrics.train_ic,
                "val_ic": metrics.val_ic,
                "test_ic": metrics.test_ic,
                "train_rank_ic": metrics.train_rank_ic,
                "val_rank_ic": metrics.val_rank_ic,
                "test_rank_ic": metrics.test_rank_ic,
                "num_graphs": len(graph_dataset.graphs),
                "num_nodes": len(graph_dataset.node_names),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    attention_paths = save_attention_heatmap(model, graph_dataset, config, config.processed_dir)

    print(f"Train RMSE: {metrics.train_rmse:.4f} | MAE: {metrics.train_mae:.4f} | RankIC: {metrics.train_rank_ic:.4f}")
    print(f"Validation RMSE: {metrics.val_rmse:.4f} | MAE: {metrics.val_mae:.4f} | RankIC: {metrics.val_rank_ic:.4f}")
    print(f"Test RMSE: {metrics.test_rmse:.4f} | MAE: {metrics.test_mae:.4f} | RankIC: {metrics.test_rank_ic:.4f}")
    print(f"Attention Heatmap: {attention_paths['png']}")

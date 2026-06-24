from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from dmrgat.config import ProjectConfig
from dmrgat.graph_builder import build_dynamic_graphs
from dmrgat.preprocess import build_feature_frame, save_feature_frame

from train_eval import fit_and_evaluate_cnn, save_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CNN baseline for sector-rotation prediction.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "default.json"),
        help="Path to the shared DMRGAT JSON config.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of daily graph snapshots per CNN batch.",
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=None,
        help="CNN hidden channels. Defaults to config.hidden_dim.",
    )
    parser.add_argument(
        "--metrics-path",
        default=str(ROOT / "CNN" / "cnn_metrics.json"),
        help="Where to save CNN evaluation metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProjectConfig.from_json(args.config)

    dataset = build_feature_frame(config)
    save_feature_frame(dataset, config)
    graph_dataset = build_dynamic_graphs(dataset, config)
    if not graph_dataset.graphs:
        raise RuntimeError("No valid samples were built. Check raw data and config date range.")

    metrics, _ = fit_and_evaluate_cnn(
        graph_dataset,
        config,
        batch_size=args.batch_size,
        hidden_channels=args.hidden_channels,
    )
    save_metrics(metrics, args.metrics_path)

    print(f"Train RMSE: {metrics.train_rmse:.4f} | MAE: {metrics.train_mae:.4f} | RankIC: {metrics.train_rank_ic:.4f}")
    print(f"Validation RMSE: {metrics.val_rmse:.4f} | MAE: {metrics.val_mae:.4f} | RankIC: {metrics.val_rank_ic:.4f}")
    print(f"Test RMSE: {metrics.test_rmse:.4f} | MAE: {metrics.test_mae:.4f} | RankIC: {metrics.test_rank_ic:.4f}")
    print(f"Metrics saved to: {args.metrics_path}")


if __name__ == "__main__":
    main()

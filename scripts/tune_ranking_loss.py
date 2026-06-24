from __future__ import annotations

import json
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dmrgat.config import ProjectConfig
from dmrgat.graph_builder import build_dynamic_graphs
from dmrgat.preprocess import build_feature_frame
from dmrgat.train_eval import fit_and_evaluate
from dmrgat.utils import ensure_dir


def make_cfg(base: ProjectConfig, ranking_loss_weight: float, mse_loss_weight: float) -> ProjectConfig:
    cfg_dict = asdict(base)
    cfg_dict["ranking_loss_weight"] = ranking_loss_weight
    cfg_dict["mse_loss_weight"] = mse_loss_weight
    return ProjectConfig(**cfg_dict)


def main() -> None:
    base = ProjectConfig.from_json(ROOT / "configs" / "default.json")
    dataset = build_feature_frame(base)
    search_base = asdict(base)
    search_base["epochs"] = 15
    search_base["patience"] = 3
    search_cfg = ProjectConfig(**search_base)

    candidates = [
        (1.0, 0.2),
        (2.0, 0.1),
        (4.0, 0.05),
    ]

    results = []
    best_cfg = None
    best_key = None
    best_val_rank_ic = float("-inf")

    for ranking_w, mse_w in candidates:
        cfg = make_cfg(search_cfg, ranking_w, mse_w)
        graph_dataset = build_dynamic_graphs(dataset, cfg)
        metrics = fit_and_evaluate(graph_dataset, cfg)
        row = {
            "ranking_loss_weight": ranking_w,
            "mse_loss_weight": mse_w,
            "train_rmse": metrics.train_rmse,
            "val_rmse": metrics.val_rmse,
            "test_rmse": metrics.test_rmse,
            "train_rank_ic": metrics.train_rank_ic,
            "val_rank_ic": metrics.val_rank_ic,
            "test_rank_ic": metrics.test_rank_ic,
        }
        results.append(row)
        print(
            f"ranking={ranking_w:.2f}, mse={mse_w:.2f}: "
            f"val_rank_ic={metrics.val_rank_ic:.4f}, test_rank_ic={metrics.test_rank_ic:.4f}, "
            f"test_rmse={metrics.test_rmse:.4f}"
        )
        if metrics.val_rank_ic > best_val_rank_ic:
            best_val_rank_ic = metrics.val_rank_ic
            best_cfg = deepcopy(make_cfg(base, ranking_w, mse_w))
            best_key = (ranking_w, mse_w)

        partial_output = {
            "search_results": results,
            "best_params_so_far": None if best_key is None else {
                "ranking_loss_weight": best_key[0],
                "mse_loss_weight": best_key[1],
            },
        }
        out_path = ensure_dir(ROOT / "data" / "processed") / "ranking_tuning_results.json"
        out_path.write_text(json.dumps(partial_output, indent=2, ensure_ascii=False), encoding="utf-8")

    if best_cfg is None or best_key is None:
        raise RuntimeError("No ranking-loss candidate succeeded.")

    final_graph_dataset = build_dynamic_graphs(dataset, best_cfg)
    final_metrics = fit_and_evaluate(final_graph_dataset, best_cfg)

    output = {
        "search_results": results,
        "best_params": {
            "ranking_loss_weight": best_key[0],
            "mse_loss_weight": best_key[1],
        },
        "final_run": {
            "train_rmse": final_metrics.train_rmse,
            "val_rmse": final_metrics.val_rmse,
            "test_rmse": final_metrics.test_rmse,
            "train_mae": final_metrics.train_mae,
            "val_mae": final_metrics.val_mae,
            "test_mae": final_metrics.test_mae,
            "train_ic": final_metrics.train_ic,
            "val_ic": final_metrics.val_ic,
            "test_ic": final_metrics.test_ic,
            "train_rank_ic": final_metrics.train_rank_ic,
            "val_rank_ic": final_metrics.val_rank_ic,
            "test_rank_ic": final_metrics.test_rank_ic,
            "num_graphs": len(final_graph_dataset.graphs),
            "num_nodes": len(final_graph_dataset.node_names),
            "ranking_loss_weight": best_key[0],
            "mse_loss_weight": best_key[1],
        },
    }

    out_path = ensure_dir(ROOT / "data" / "processed") / "ranking_tuning_results.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Best params: ranking={best_key[0]:.2f}, mse={best_key[1]:.2f}; "
        f"final test RankIC={final_metrics.test_rank_ic:.4f}"
    )


if __name__ == "__main__":
    main()

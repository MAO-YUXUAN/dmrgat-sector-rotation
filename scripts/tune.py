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


def make_candidate(base: ProjectConfig, name: str, **overrides) -> tuple[str, ProjectConfig]:
    cfg_dict = asdict(base)
    cfg_dict.update(overrides)
    return name, ProjectConfig(**cfg_dict)


def main() -> None:
    base = ProjectConfig.from_json(ROOT / "configs" / "default.json")
    search_epochs = 35
    search_patience = 6

    candidates = [
        make_candidate(base, "baseline_fast", epochs=search_epochs, patience=search_patience),
        make_candidate(
            base,
            "graph_focus_1",
            alpha=0.60,
            beta=0.10,
            gamma=0.30,
            top_k_neighbors=5,
            epochs=search_epochs,
            patience=search_patience,
        ),
        make_candidate(
            base,
            "graph_focus_2",
            alpha=0.55,
            beta=0.15,
            gamma=0.30,
            top_k_neighbors=10,
            epochs=search_epochs,
            patience=search_patience,
        ),
        make_candidate(
            base,
            "graph_focus_3",
            alpha=0.45,
            beta=0.10,
            gamma=0.45,
            top_k_neighbors=8,
            corr_window=30,
            epochs=search_epochs,
            patience=search_patience,
        ),
        make_candidate(
            base,
            "model_focus_1",
            hidden_dim=64,
            num_heads=4,
            dropout=0.30,
            learning_rate=5e-4,
            epochs=search_epochs,
            patience=search_patience,
        ),
        make_candidate(
            base,
            "model_focus_2",
            hidden_dim=64,
            num_heads=2,
            dropout=0.35,
            learning_rate=5e-4,
            weight_decay=5e-4,
            epochs=search_epochs,
            patience=search_patience,
        ),
        make_candidate(
            base,
            "lookback_mix_1",
            feature_lookbacks=[3, 5, 10],
            corr_window=15,
            top_k_neighbors=8,
            epochs=search_epochs,
            patience=search_patience,
        ),
        make_candidate(
            base,
            "lookback_mix_2",
            feature_lookbacks=[5, 10, 30],
            corr_window=30,
            alpha=0.55,
            beta=0.10,
            gamma=0.35,
            top_k_neighbors=10,
            hidden_dim=64,
            num_heads=4,
            dropout=0.30,
            learning_rate=5e-4,
            epochs=search_epochs,
            patience=search_patience,
        ),
    ]

    results = []
    best_name = None
    best_cfg = None
    best_val = float("-inf")

    for name, cfg in candidates:
        dataset = build_feature_frame(cfg)
        graph_dataset = build_dynamic_graphs(dataset, cfg)
        metrics = fit_and_evaluate(graph_dataset, cfg)
        row = {
            "name": name,
            "train_accuracy": metrics.train_accuracy,
            "val_accuracy": metrics.val_accuracy,
            "test_accuracy": metrics.test_accuracy,
            "num_graphs": len(graph_dataset.graphs),
            "params": {
                "horizon": cfg.horizon,
                "corr_window": cfg.corr_window,
                "feature_lookbacks": cfg.feature_lookbacks,
                "alpha": cfg.alpha,
                "beta": cfg.beta,
                "gamma": cfg.gamma,
                "top_k_neighbors": cfg.top_k_neighbors,
                "hidden_dim": cfg.hidden_dim,
                "num_heads": cfg.num_heads,
                "dropout": cfg.dropout,
                "learning_rate": cfg.learning_rate,
                "weight_decay": cfg.weight_decay,
                "epochs": cfg.epochs,
                "patience": cfg.patience,
            },
        }
        results.append(row)
        if metrics.val_accuracy > best_val:
            best_val = metrics.val_accuracy
            best_name = name
            best_cfg = deepcopy(cfg)

        print(
            f"{name}: train={metrics.train_accuracy:.4f}, "
            f"val={metrics.val_accuracy:.4f}, test={metrics.test_accuracy:.4f}"
        )

    if best_cfg is None or best_name is None:
        raise RuntimeError("No candidate finished successfully.")

    final_cfg_dict = asdict(best_cfg)
    final_cfg_dict["epochs"] = 80
    final_cfg_dict["patience"] = 12
    final_cfg = ProjectConfig(**final_cfg_dict)

    final_dataset = build_feature_frame(final_cfg)
    final_graph_dataset = build_dynamic_graphs(final_dataset, final_cfg)
    final_metrics = fit_and_evaluate(final_graph_dataset, final_cfg)

    output = {
        "search_results": results,
        "best_search_name": best_name,
        "best_search_val_accuracy": best_val,
        "best_search_params": {
            "horizon": best_cfg.horizon,
            "corr_window": best_cfg.corr_window,
            "feature_lookbacks": best_cfg.feature_lookbacks,
            "alpha": best_cfg.alpha,
            "beta": best_cfg.beta,
            "gamma": best_cfg.gamma,
            "top_k_neighbors": best_cfg.top_k_neighbors,
            "hidden_dim": best_cfg.hidden_dim,
            "num_heads": best_cfg.num_heads,
            "dropout": best_cfg.dropout,
            "learning_rate": best_cfg.learning_rate,
            "weight_decay": best_cfg.weight_decay,
        },
        "final_run": {
            "train_accuracy": final_metrics.train_accuracy,
            "val_accuracy": final_metrics.val_accuracy,
            "test_accuracy": final_metrics.test_accuracy,
            "num_graphs": len(final_graph_dataset.graphs),
            "num_nodes": len(final_graph_dataset.node_names),
            "params": {
                "horizon": final_cfg.horizon,
                "corr_window": final_cfg.corr_window,
                "feature_lookbacks": final_cfg.feature_lookbacks,
                "alpha": final_cfg.alpha,
                "beta": final_cfg.beta,
                "gamma": final_cfg.gamma,
                "top_k_neighbors": final_cfg.top_k_neighbors,
                "hidden_dim": final_cfg.hidden_dim,
                "num_heads": final_cfg.num_heads,
                "dropout": final_cfg.dropout,
                "learning_rate": final_cfg.learning_rate,
                "weight_decay": final_cfg.weight_decay,
                "epochs": final_cfg.epochs,
                "patience": final_cfg.patience,
            },
        },
    }

    processed_dir = ensure_dir(ROOT / "data" / "processed")
    output_path = processed_dir / "tuning_results.json"
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Best search candidate: {best_name}")
    print(
        f"Final run: train={final_metrics.train_accuracy:.4f}, "
        f"val={final_metrics.val_accuracy:.4f}, test={final_metrics.test_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()

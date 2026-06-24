from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import torch

from .graph_builder import GraphDataset, split_graphs
from .model import DMRGAT
from .utils import ensure_dir, get_device


def _get_chinese_font() -> font_manager.FontProperties | None:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "PingFang SC",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["axes.unicode_minus"] = False
            return font_manager.FontProperties(family=name)
    plt.rcParams["axes.unicode_minus"] = False
    return None


def _attention_to_matrix(
    edge_index: torch.Tensor,
    alpha: torch.Tensor,
    node_names: list[str],
) -> list[pd.DataFrame]:
    src = edge_index[0].detach().cpu().numpy()
    dst = edge_index[1].detach().cpu().numpy()
    attn = alpha.detach().cpu().numpy()
    if attn.ndim == 1:
        attn = attn[:, None]

    matrices: list[pd.DataFrame] = []
    for head_idx in range(attn.shape[1]):
        matrix = np.zeros((len(node_names), len(node_names)), dtype=float)
        for s, d, a in zip(src, dst, attn[:, head_idx], strict=False):
            matrix[d, s] = float(a)
        matrices.append(pd.DataFrame(matrix, index=node_names, columns=node_names))
    return matrices


def _save_heatmap_figure(heatmap_df: pd.DataFrame, title: str, png_path: Path, font_prop) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(heatmap_df.to_numpy(), cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(heatmap_df.columns)))
    ax.set_yticks(range(len(heatmap_df.index)))
    ax.set_xticklabels(heatmap_df.columns, rotation=45, ha="right", fontsize=9, fontproperties=font_prop)
    ax.set_yticklabels(heatmap_df.index, fontsize=9, fontproperties=font_prop)
    ax.set_title(title, fontproperties=font_prop)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Attention Weight")
    fig.tight_layout()
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def save_attention_heatmap(
    model: DMRGAT,
    graph_dataset: GraphDataset,
    config,
    output_dir: str | Path,
) -> dict[str, str]:
    splits = split_graphs(graph_dataset, config)
    graph = splits["test"][-1] if splits["test"] else graph_dataset.graphs[-1]
    graph_date = getattr(graph, "trade_date", "unknown_date")

    device = get_device(config.use_gpu)
    model = model.to(device)
    model.eval()
    graph = graph.to(device)

    attention_maps = model.get_attention_maps(graph.x, graph.edge_index, graph.edge_attr)
    edge_index, alpha = attention_maps["layer1"]
    head_heatmaps = _attention_to_matrix(edge_index, alpha, graph_dataset.node_names)
    avg_heatmap = sum(head_heatmaps) / len(head_heatmaps)

    out_dir = ensure_dir(output_dir)
    csv_path = out_dir / "attention_heatmap.csv"
    png_path = out_dir / "attention_heatmap.png"
    meta_path = out_dir / "attention_heatmap_meta.json"

    avg_heatmap.to_csv(csv_path, encoding="utf-8-sig")

    font_prop = _get_chinese_font()
    _save_heatmap_figure(
        avg_heatmap,
        f"DMRGAT Layer1 Attention Heatmap Avg ({graph_date})",
        png_path,
        font_prop,
    )

    head_paths: dict[str, str] = {}
    for head_idx, head_df in enumerate(head_heatmaps, start=1):
        head_csv = out_dir / f"attention_heatmap_head_{head_idx}.csv"
        head_png = out_dir / f"attention_heatmap_head_{head_idx}.png"
        head_df.to_csv(head_csv, encoding="utf-8-sig")
        _save_heatmap_figure(
            head_df,
            f"DMRGAT Layer1 Attention Head {head_idx} ({graph_date})",
            head_png,
            font_prop,
        )
        head_paths[f"head_{head_idx}_csv"] = str(head_csv)
        head_paths[f"head_{head_idx}_png"] = str(head_png)

    meta_path.write_text(
        pd.Series(
            {
                "trade_date": graph_date,
                "num_nodes": len(graph_dataset.node_names),
                "source_split": "test" if splits["test"] else "all",
                "layer": "layer1",
                "num_heads": len(head_heatmaps),
            }
        ).to_json(force_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "csv": str(csv_path),
        "png": str(png_path),
        "meta": str(meta_path),
        **head_paths,
    }

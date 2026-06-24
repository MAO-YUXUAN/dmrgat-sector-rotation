from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ProjectConfig:
    tushare_token: str = ""
    start_date: str = "20160101"
    end_date: str = "20251231"
    benchmark_code: str = "000300.SH"
    horizon: int = 20
    corr_window: int = 20
    feature_lookbacks: list[int] = field(default_factory=lambda: [5, 10, 20])
    alpha: float = 0.45
    beta: float = 0.25
    gamma: float = 0.30
    top_k_neighbors: int = 8
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    hidden_dim: int = 32
    num_heads: int = 4
    dropout: float = 0.20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 80
    patience: int = 12
    ranking_loss_weight: float = 1.0
    mse_loss_weight: float = 0.2
    seed: int = 42
    use_gpu: bool = True
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    label_epsilon: float = 0.01
    core_industries: list[str] = field(
        default_factory=lambda: [
            "农林牧渔",
            "基础化工",
            "有色金属",
            "钢铁",
            "电力设备",
            "机械设备",
            "汽车",
            "电子",
            "计算机",
            "通信",
            "医药生物",
            "食品饮料",
            "银行",
            "非银金融",
            "交通运输",
        ]
    )

    @classmethod
    def from_json(cls, path: str | Path) -> "ProjectConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

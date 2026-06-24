from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dmrgat.config import ProjectConfig
from dmrgat.pipeline import run_download, run_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DMRGAT for CITIC level-1 industry rotation prediction.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "default.json"),
        help="Path to JSON config file.",
    )
    parser.add_argument(
        "--stage",
        choices=["download", "train", "all"],
        default="all",
        help="Pipeline stage to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProjectConfig.from_json(args.config)

    if args.stage in {"download", "all"}:
        run_download(config)
    if args.stage in {"train", "all"}:
        run_train(config)


if __name__ == "__main__":
    main()


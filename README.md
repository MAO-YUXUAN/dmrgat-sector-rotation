# DMRGAT Sector Rotation Forecasting

This repository implements a Dynamic Multi-Relation Graph Attention Network (DMRGAT) for industry-level sector rotation forecasting in the Chinese A-share market.

The project predicts each Shenwan level-1 industry's future 30-trading-day excess return over the CSI 300 benchmark. It models industries as graph nodes and constructs dynamic multi-relation graphs from return co-movement, trading-amount similarity, and static industry-prior relations.

## Highlights

- Data pipeline based on AKShare.
- Shenwan level-1 industry universe with 15 core industries.
- Future 30-day benchmark-relative excess return prediction.
- Dynamic graph construction:
  - rolling return-correlation graph;
  - rolling trading-amount similarity graph as a capital-flow proxy;
  - static economic industry-prior graph.
- Edge-weighted graph attention:

```text
e_ij,t = A_ij,t * LeakyReLU(a^T [z_i || z_j])
```

- Ranking-oriented training objective:

```text
L = 4.0 * L_rank + 0.05 * L_mse
```

- CNN and LSTM baselines for comparison.
- Attention heatmap export for model interpretation.
- Overleaf-ready LaTeX paper in `overleaf_dmrgat/`.

## Repository Structure

```text
.
├── CNN/                  # CNN non-graph baseline
├── LSTM/                 # LSTM time-series baseline
├── configs/              # JSON configuration
├── overleaf_dmrgat/      # LaTeX paper source
├── results/              # lightweight result summary
├── scripts/              # runnable scripts and document-generation helpers
├── src/dmrgat/           # DMRGAT package
├── requirements.txt
└── README.md
```

Large raw datasets, processed feature files, Word manuscripts, PDFs, and generated caches are intentionally excluded from Git.

## Installation

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

PyTorch Geometric installation may depend on your local PyTorch/CUDA version. If installation fails, follow the official PyTorch Geometric wheel instructions for your environment.

## Configuration

Edit `configs/default.json` to change:

- date range;
- industry universe;
- benchmark code;
- forecasting horizon;
- graph fusion coefficients;
- model hyperparameters;
- train/validation/test split.

Current default setting:

```text
horizon = 30
corr_window = 20
alpha = 0.45
beta = 0.25
gamma = 0.30
train/val/test = 80% / 10% / 10%
```

## Run DMRGAT

Download data only:

```bash
python scripts/main.py --stage download
```

Download data and train:

```bash
python scripts/main.py --stage all
```

Train from existing raw data:

```bash
python scripts/main.py --stage train
```

The main outputs are saved under `data/processed/`:

- `metrics.json`;
- `features.csv`;
- `attention_heatmap.csv`;
- `attention_heatmap.png`;
- per-head attention heatmaps.

## Run Baselines

CNN baseline:

```bash
python CNN/main.py
```

LSTM baseline:

```bash
python LSTM/main.py
```

Both baselines reuse the same preprocessing pipeline, feature definitions, target variable, and chronological split as DMRGAT.

## Current Results

See [results/README.md](results/README.md) for the latest metrics.

Summary of the current test-set results:

| Model | Test RMSE | Test MAE | Test RankIC |
|---|---:|---:|---:|
| DMRGAT | 0.1829 | 0.1660 | 0.0122 |
| CNN | 0.1645 | 0.1282 | -0.0266 |
| LSTM | 0.2582 | 0.2022 | 0.0082 |

The CNN baseline achieves lower point-forecast error, while DMRGAT produces a weak but positive test RankIC, which is more aligned with ranking-based sector allocation.

## Paper

An Overleaf-ready LaTeX manuscript is available in:

```text
overleaf_dmrgat/
```

Upload the folder to Overleaf and compile `main.tex`.

## Notes

- This project is a research prototype, not a trading system.
- The model does not include transaction-cost-aware portfolio backtesting.
- Attention heatmaps should be interpreted as model diagnostics, not causal evidence.
- AKShare data access may change if upstream data providers update their endpoints.

## License

This project is released under the MIT License.

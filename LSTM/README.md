# LSTM Baseline for Sector-Rotation Prediction

This folder implements an LSTM baseline for the same prediction target used by the DMRGAT and CNN models.

## Purpose

The LSTM baseline predicts each Shenwan level-1 industry's future 30-trading-day excess return over the CSI 300 benchmark. It reuses the same preprocessing pipeline, node features, target definition, and chronological train/validation/test split style.

Unlike DMRGAT, it does not use graph edges or attention. Unlike CNN, it explicitly uses a historical feature sequence. For each sample, the default input shape is:

```text
seq_len x num_industries x num_features
20 x 15 x 15
```

The model applies the same LSTM to each industry node's historical feature sequence and outputs one predicted future excess return per industry.

## Files

- `main.py`: command-line entry point.
- `model.py`: LSTM model definition.
- `train_eval.py`: sequence dataset, training loop, ranking-regression loss, evaluation metrics, and metric saving.
- `lstm_metrics.json`: generated after running training.

## Run

From the project root:

```powershell
& 'C:\Users\admin\AppData\Local\Programs\Python\Python311\python.exe' LSTM/main.py
```

Optional arguments:

```powershell
& 'C:\Users\admin\AppData\Local\Programs\Python\Python311\python.exe' LSTM/main.py --seq-len 20 --batch-size 32 --hidden-dim 32 --num-layers 1
```

## Output Metrics

The script prints and saves:

- Train/validation/test RMSE
- Train/validation/test MAE
- Train/validation/test IC
- Train/validation/test RankIC

The default output file is:

```text
LSTM/lstm_metrics.json
```

## Comparison

Use:

- `data/processed/metrics.json` for DMRGAT
- `CNN/cnn_metrics.json` for CNN
- `LSTM/lstm_metrics.json` for LSTM

The three models share the same feature and target definitions, making the comparison suitable for discussing graph-based modeling, convolutional baselines, and recurrent time-series baselines.

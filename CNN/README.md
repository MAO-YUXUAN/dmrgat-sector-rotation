# CNN Baseline for Sector-Rotation Prediction

This folder implements a convolutional neural network baseline for the same prediction task used by the DMRGAT model.

## Purpose

The CNN baseline predicts each Shenwan level-1 industry's future 30-trading-day excess return over the CSI 300 benchmark. It uses the same data pipeline, feature matrix, target definition, chronological train/validation/test split, and ranking-oriented loss as the DMRGAT implementation.

The main difference is that CNN does not use graph edges or attention. For each trading date, it treats the industry-feature matrix as a small 2D input:

```text
num_industries x num_features
```

This makes it useful as a non-graph deep-learning baseline for later CNN vs. GAT comparison.

## Files

- `main.py`: command-line entry point for training and evaluating the CNN baseline.
- `model.py`: CNN model definition.
- `train_eval.py`: training loop, ranking-regression loss, evaluation metrics, and metric saving.
- `cnn_metrics.json`: generated after running training.

## Run

From the project root:

```powershell
& 'C:\Users\admin\AppData\Local\Programs\Python\Python311\python.exe' CNN/main.py
```

Optional arguments:

```powershell
& 'C:\Users\admin\AppData\Local\Programs\Python\Python311\python.exe' CNN/main.py --batch-size 32 --hidden-channels 32
```

## Output Metrics

The script prints and saves:

- Train/validation/test RMSE
- Train/validation/test MAE
- Train/validation/test IC
- Train/validation/test RankIC

The default output file is:

```text
CNN/cnn_metrics.json
```

## Comparison with DMRGAT

Use `data/processed/metrics.json` for DMRGAT results and `CNN/cnn_metrics.json` for CNN results. Since both models share the same feature construction and split logic, their metrics are directly comparable at the modeling layer.

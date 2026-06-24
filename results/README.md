# Result Summary

The current experiment uses:

- 15 core Shenwan level-1 industries;
- CSI 300 as benchmark;
- 30-trading-day future excess return target;
- 20-trading-day rolling graph window;
- chronological 80/10/10 train-validation-test split;
- ranking-dominant loss for DMRGAT.

## DMRGAT

| Split | RMSE | MAE | IC | RankIC |
|---|---:|---:|---:|---:|
| Train | 0.1650 | 0.1459 | 0.0278 | 0.0344 |
| Validation | 0.1682 | 0.1422 | 0.0722 | 0.0897 |
| Test | 0.1829 | 0.1660 | 0.0139 | 0.0122 |

## CNN Baseline

| Split | RMSE | MAE | IC | RankIC |
|---|---:|---:|---:|---:|
| Train | 0.1535 | 0.1194 | 0.1695 | 0.1618 |
| Validation | 0.1675 | 0.1269 | 0.0854 | 0.0862 |
| Test | 0.1645 | 0.1282 | -0.0257 | -0.0266 |

## LSTM Baseline

| Split | RMSE | MAE | IC | RankIC |
|---|---:|---:|---:|---:|
| Train | 0.2817 | 0.2233 | 0.1735 | 0.1633 |
| Validation | 0.4161 | 0.3405 | 0.1476 | 0.1770 |
| Test | 0.2582 | 0.2022 | 0.0163 | 0.0082 |

## Interpretation

CNN has the lowest test RMSE and MAE but a negative test RankIC. DMRGAT does not minimize point-forecast error, but it preserves a weak positive test ranking signal. This distinction matters because sector rotation is primarily a cross-sectional ranking problem.

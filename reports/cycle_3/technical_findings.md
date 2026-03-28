# Technical Findings — Cycle 3 (Phase 3: Walk-Forward Evaluation Framework)

## Implementation Summary

Implemented a complete walk-forward evaluation framework for the LSTM portfolio optimization model, including:

1. **Data Pipeline** (`src/data_pipeline.py`): Fetches daily OHLCV data for 10 ETFs from the ARF Data API with local caching. Computes returns and prepares rolling-window sequences for the LSTM.

2. **LSTM Model** (`src/model.py`): 2-layer LSTM (hidden_size=64) with softmax output producing long-only portfolio weights. Trained with a custom negative Sharpe ratio loss function via Adam optimizer.

3. **Evaluation Framework** (`src/evaluation.py`): `WalkForwardEvaluator` class orchestrating 5-fold expanding-window walk-forward validation. Each fold trains a fresh model on historical data and evaluates on the subsequent unseen period.

4. **CLI** (`src/cli.py`): `run-backtest` command that executes the full pipeline and outputs results to `reports/cycle_3/`.

## Results (from metrics.json)

### Model (LSTM + Sharpe Loss)
| Metric | Value |
|---|---|
| Gross Sharpe (avg) | 0.9471 |
| Net Sharpe (avg) | 0.8684 |
| Annual Return | 12.96% |
| Max Drawdown | -14.65% |
| Hit Rate | 55.0% |
| Positive Windows | 4/5 |
| Total Trades | 649 |

### Baseline (1/N Equal Weight)
| Metric | Value |
|---|---|
| Sharpe (avg) | 1.1041 |
| Annual Return | 11.65% |
| Max Drawdown | -14.89% |
| Hit Rate | 53.51% |
| Positive Windows | 5/5 |

### Per-Window Comparison
| Window | Period | Model Net Sharpe | 1/N Sharpe | Model Wins? |
|---|---|---|---|---|
| 1 | 2022-06 to 2023-03 | -0.6645 | 0.1384 | No |
| 2 | 2023-03 to 2023-12 | 0.7106 | 1.4771 | No |
| 3 | 2023-12 to 2024-09 | 1.7833 | 1.6275 | Yes |
| 4 | 2024-09 to 2025-06 | 0.6650 | 0.6331 | Yes |
| 5 | 2025-06 to 2026-03 | 1.8474 | 1.6444 | Yes |

## Observations

1. **Model vs Baseline**: The LSTM model's average net Sharpe (0.87) is below the 1/N baseline (1.10). However, the model outperforms in 3 of 5 windows, with increasing performance in later windows as training data grows.

2. **Window 1 Underperformance**: The first window (2022-06 to 2023-03) shows negative performance, coinciding with the 2022 bear market. With only ~254 training samples, the model had insufficient data to learn robust patterns.

3. **Improving with Data**: Later windows (3-5) consistently outperform the baseline, suggesting the model benefits from expanding training windows. This is consistent with deep learning models requiring sufficient data.

4. **Transaction Costs**: The gap between gross Sharpe (0.95) and net Sharpe (0.87) shows ~8bp impact from the 15bps total cost model (10bps fee + 5bps slippage).

5. **No Data Leakage**: Verified through:
   - Scaler fit on train data only per fold
   - Gap of 1 day between train and test
   - No overlap between train/test indices
   - Features use only t-lookback to t-1 data for prediction at time t

## Transaction Cost Configuration
- Fee: 10 bps
- Slippage: 5 bps
- Total: 15 bps per trade

## Data Leakage Prevention
- StandardScaler fit only on training fold data
- Walk-forward gap=1 between train and test periods
- Rolling window features use `center=False` (no future data)
- Expanding window training: each fold uses all prior data

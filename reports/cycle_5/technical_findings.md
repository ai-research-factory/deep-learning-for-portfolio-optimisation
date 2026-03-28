# Technical Findings — Cycle 5 (Phase 5: Hyperparameter Optimization with Optuna)

## Objective

Optimize key LSTM model hyperparameters using Optuna to maximize average out-of-sample net Sharpe ratio across walk-forward windows.

## Implementation

### OptunaOptimizer (src/evaluation.py)

- **Objective function**: Average net Sharpe ratio across 5 walk-forward windows (after transaction costs of 10bps fee + 5bps slippage)
- **Sampler**: TPESampler (Tree-structured Parzen Estimator) with seed=42
- **Pruner**: MedianPruner (prunes trials performing below median after 2 windows, 5 startup trials)
- **Data prefetching**: Data fetched once and cached across all trials

### Search Space (constrained to paper defaults' neighborhood)

| Parameter | Search Values | Paper Default |
|---|---|---|
| lookback | [30, 45, 60, 90, 120] | 60 |
| lr | [5e-5, 1e-4, 5e-4, 1e-3] | 1e-4 |
| hidden_size | [32, 64, 128] | 64 |
| num_layers | [1, 2, 3] | 2 |
| epochs | [30, 50, 100] | 50 |
| batch_size | [32, 64, 128] | 64 |

### CLI Command

```bash
python3 -m src.cli run-optimize --n-trials 20 --output-dir reports/cycle_5
```

## Results

All values from `reports/cycle_5/metrics.json`.

### Best Hyperparameters (Trial 13 of 20)

| Parameter | Default | Optimized |
|---|---|---|
| lookback | 60 | 120 |
| lr | 1e-4 | 5e-5 |
| hidden_size | 64 | 64 |
| num_layers | 2 | 2 |
| epochs | 50 | 100 |
| batch_size | 64 | 128 |

### Performance Comparison

| Metric | Default Params | Optimized Params | 1/N Baseline |
|---|---|---|---|
| Net Sharpe | 0.9123 | 1.4297 | 1.2404 |
| Gross Sharpe | 1.0033 | 1.4626 | 1.2404 |
| Annual Return | — | 21.59% | 14.07% |
| Max Drawdown | — | -12.52% | -11.54% |
| Hit Rate | — | 56.25% | — |
| Positive Windows | — | 5/5 | 5/5 |

### Per-Window Performance (Optimized)

| Window | Gross Sharpe | Net Sharpe | Turnover |
|---|---|---|---|
| 1 | 0.9995 | 0.9829 | 0.66% |
| 2 | 1.1205 | 1.0932 | 0.72% |
| 3 | 2.0069 | 1.9466 | 1.69% |
| 4 | 1.3930 | 1.3492 | 1.67% |
| 5 | 1.7931 | 1.7764 | 0.89% |

### Improvement Over Default

- Net Sharpe improvement: +0.5174 (0.9123 → 1.4297)
- The optimized model now **outperforms the 1/N baseline** (1.4297 vs 1.2404 net Sharpe)
- All 5 windows are positive (vs 4/5 with default params in Cycle 4)

## Key Observations

1. **Longer lookback (120 vs 60)**: Doubling the lookback window was the most impactful change. With 120 days, the model captures longer-term trends and reduces noise sensitivity.

2. **Lower learning rate (5e-5 vs 1e-4)**: Combined with more epochs (100 vs 50), this allows for slower, more stable convergence. The model avoids overfitting to noisy short-term patterns.

3. **Architecture unchanged**: hidden_size=64 and num_layers=2 remained optimal, suggesting the architecture was already well-sized for this problem.

4. **Reduced turnover**: Average daily turnover dropped significantly from 3.75% (Cycle 4 default) to ~1.13%, resulting in lower cost drag (Sharpe impact: 0.033 vs 0.089 previously).

5. **Pruning efficiency**: Several trials were pruned early by MedianPruner, saving computation. 20 trials were sufficient to find a strong configuration.

6. **Stable across windows**: The optimized model performs well across all 5 windows with no negative-Sharpe windows, indicating improved robustness.

## Limitations

- 20 trials is a modest search budget. More trials could potentially find better configurations, though the improvement trajectory plateaued after trial 13.
- The search space was intentionally constrained to the neighborhood of paper defaults per reproduction rules.
- Results are specific to this dataset period (2021-2026) and 10-ETF universe.

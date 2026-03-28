# Technical Findings — Cycle 4 (Phase 4: Transaction Cost Model)

## Implementation Summary

### What was implemented
1. **`TransactionCostModel` class** (`src/backtest.py`): A dedicated cost model that computes turnover from portfolio weight changes and applies proportional fee and slippage costs separately. Provides detailed `CostBreakdown` per window (total cost, fee cost, slippage cost, turnover statistics).

2. **Cost sensitivity analysis** (`src/evaluation.py`): `run_cost_sensitivity()` method that evaluates model performance across multiple cost levels (0, 5, 10, 15, 20, 30 bps) using a single set of trained models — costs are applied post-hoc to avoid retraining.

3. **Gross vs net comparison**: Extended `BacktestResult` to track both `gross_sharpe` and `gross_annual_return` alongside net metrics, enabling direct comparison.

4. **CLI commands** (`src/cli.py`):
   - `run-backtest`: Updated with `--fee-bps` and `--slippage-bps` options, outputs gross/net comparison.
   - `run-cost-analysis`: New command for cost sensitivity analysis with `--cost-levels` option.

5. **Tests**: 7 new tests for `TransactionCostModel` covering zero-cost identity, cost reduction, monotonicity, turnover computation, breakdown correctness, config construction, and no-rebalance identity.

## Results

All values from `reports/cycle_4/metrics.json`.

### Primary Backtest (15bps total cost: 10bps fee + 5bps slippage)

| Metric | LSTM Model (Gross) | LSTM Model (Net) | 1/N Baseline |
|---|---|---|---|
| Avg OOS Sharpe | 0.7052 | 0.6161 | 1.1041 |
| Annual Return | — | 7.16% | 11.65% |
| Max Drawdown | — | -16.93% | -14.89% |
| Hit Rate | — | 53.83% | 53.51% |
| Positive Windows | — | 4/5 | 5/5 |
| Cost Impact (Sharpe) | — | -0.0891 | 0.0 |

### Per-Window Breakdown

| Window | Period | Gross Sharpe | Net Sharpe | 1/N Sharpe | Avg Daily Turnover |
|---|---|---|---|---|---|
| 1 | 2022-06 to 2023-03 | -0.6202 | -0.6427 | 0.1384 | 1.07% |
| 2 | 2023-03 to 2023-12 | 0.7444 | 0.7058 | 1.4771 | 1.05% |
| 3 | 2023-12 to 2024-09 | 1.8116 | 1.6728 | 1.6275 | 3.95% |
| 4 | 2024-09 to 2025-06 | 0.1543 | 0.0577 | 0.6331 | 4.90% |
| 5 | 2025-06 to 2026-03 | 1.4358 | 1.2871 | 1.6444 | 6.79% |

### Cost Sensitivity Analysis

| Cost (bps) | Model Net Sharpe | 1/N Net Sharpe | Delta |
|---|---|---|---|
| 0 | 1.0844 | 1.1041 | -0.0197 |
| 5 | 1.0543 | 1.1041 | -0.0498 |
| 10 | 1.0243 | 1.1041 | -0.0798 |
| 15 | 0.9942 | 1.1041 | -0.1099 |
| 20 | 0.9642 | 1.1041 | -0.1399 |
| 30 | 0.9041 | 1.1041 | -0.2000 |

## Key Observations

### 1. Transaction Cost Impact
- The cost impact is **0.0891 Sharpe** at the default 15bps level (gross 0.7052 → net 0.6161).
- The model's average daily turnover is **3.75%**, meaning roughly 3.75% of portfolio value is traded per day.
- Turnover increases with more training data: Window 1 (1.07%) → Window 5 (6.79%). This suggests the model becomes more active as it learns from more data.

### 2. Cost Sensitivity
- Even at **0bps** (no costs), the model (1.0844) slightly underperforms the 1/N baseline (1.1041).
- The gap widens linearly with costs: approximately **-0.006 Sharpe per bps** of total cost.
- At all tested cost levels, the 1/N baseline outperforms the model. The baseline incurs zero rebalancing costs since equal weights are maintained passively.

### 3. Gross vs Net Performance Gap
- The gross-to-net degradation is most severe in high-turnover windows (Window 5: 1.4358 → 1.2871, delta = -0.149).
- Low-turnover windows show minimal impact (Window 1: -0.6202 → -0.6427, delta = -0.023).

### 4. Implications for Future Phases
- **Turnover regularization** could help: adding a turnover penalty to the loss function would reduce trading frequency and narrow the gross-net gap.
- The model's increasing turnover with more data suggests potential overfitting to recent patterns.
- Hyperparameter optimization (Phase 5) should consider net Sharpe as the objective, not gross.

## Data Leakage Verification
- Scaler is fit only on training data per window (no test data leakage).
- 1-day gap maintained between train and test periods.
- Transaction costs are applied to out-of-sample returns only.
- Cost sensitivity analysis reuses trained models without re-fitting (no cost information leaks into training).

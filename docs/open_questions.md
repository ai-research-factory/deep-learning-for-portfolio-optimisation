# Open Questions

## Data Constraints
- The ARF Data API returns ~5 years of daily data. The paper's exact data period is not specified, so we used the maximum available.
- DBC (commodities ETF) data availability may differ from the paper's original dataset.

## Model vs Baseline Performance
- The LSTM model underperforms the 1/N baseline on average Sharpe (0.62 net vs 1.10), though it wins in 2/5 windows (gross basis).
- Even at zero transaction costs, the model (1.08 Sharpe) slightly underperforms 1/N (1.10), indicating the performance gap is primarily due to alpha generation, not cost drag.
- The model shows improving performance with more training data (later windows), suggesting hyperparameter tuning or longer training history may help.

## Transaction Cost Model (Phase 4)
- The paper does not specify transaction cost assumptions. We use 10bps fee + 5bps slippage (15bps total) as standard institutional assumptions.
- Cost sensitivity analysis shows the model's Sharpe degrades by approximately 0.006 per bps of total cost.
- Average daily turnover is 3.75%, increasing from 1.07% (Window 1) to 6.79% (Window 5). This increasing turnover may indicate the model is learning more active strategies with more data, or potentially overfitting to recent patterns.
- Turnover regularization (adding a turnover penalty to the loss function) could improve net performance but is outside the scope of the paper's specification.

## Paper Specification Gaps
- The paper does not specify exact hyperparameters (hidden size, learning rate, epochs). Current defaults (hidden=64, lr=1e-4, epochs=50) are reasonable starting points.
- Exact walk-forward window count is not specified in the paper; we use n_splits=5 per Phase 3 task instructions.

## Future Improvements (Phase 5+)
- Hyperparameter optimization via Optuna (Phase 5) — should optimize for net Sharpe, not gross
- Alternative architectures — CNN (Phase 7)
- Turnover regularization could narrow the gross-net gap

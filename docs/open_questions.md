# Open Questions

## Data Constraints
- The ARF Data API returns ~5 years of daily data. The paper's exact data period is not specified, so we used the maximum available.
- DBC (commodities ETF) data availability may differ from the paper's original dataset.

## Model vs Baseline Performance
- The LSTM model underperforms the 1/N baseline on average Sharpe (0.87 vs 1.10), though it wins in 3/5 windows. This is consistent with literature showing the difficulty of beating equal-weight portfolios in practice.
- The model shows improving performance with more training data (later windows), suggesting hyperparameter tuning or longer training history may help.

## Paper Specification Gaps
- The paper does not specify exact hyperparameters (hidden size, learning rate, epochs). Current defaults (hidden=64, lr=1e-4, epochs=50) are reasonable starting points.
- The paper does not specify transaction cost assumptions. We use 10bps fee + 5bps slippage as standard institutional assumptions.
- Exact walk-forward window count is not specified in the paper; we use n_splits=5 per Phase 3 task instructions.

## Future Improvements (Phase 4+)
- Transaction cost sensitivity analysis (Phase 4)
- Hyperparameter optimization via Optuna (Phase 5)
- Alternative architectures — CNN (Phase 7)

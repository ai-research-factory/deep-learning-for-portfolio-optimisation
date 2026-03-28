"""
Walk-forward evaluation framework for portfolio optimization.
Phase 3: Implements walk-forward validation, backtesting, and baseline comparisons.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from src.backtest import (
    BacktestConfig,
    BacktestResult,
    WalkForwardValidator,
    calculate_costs,
    compute_metrics,
    generate_metrics_json,
)
from src.data_pipeline import (
    fetch_universe,
    compute_returns,
    prepare_sequences,
)
from src.model import train_model, predict_weights


class WalkForwardEvaluator:
    """Orchestrates walk-forward evaluation of the portfolio model vs baselines."""

    def __init__(
        self,
        tickers: list[str] | None = None,
        lookback: int = 60,
        n_splits: int = 5,
        hidden_size: int = 64,
        num_layers: int = 2,
        lr: float = 1e-4,
        epochs: int = 50,
        batch_size: int = 64,
        fee_bps: float = 10.0,
        slippage_bps: float = 5.0,
    ):
        self.tickers = tickers
        self.lookback = lookback
        self.n_splits = n_splits
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        self.config = BacktestConfig(
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            n_splits=n_splits,
            gap=1,
            min_train_size=max(252, lookback + 60),
            train_ratio=1.0,
        )

    def run(self, verbose: bool = True) -> dict:
        """Run the full walk-forward backtest.

        Returns:
            Dict with 'model' and 'baseline' results and metrics.
        """
        if verbose:
            print("Fetching data...")
        prices = fetch_universe(self.tickers)
        returns = compute_returns(prices)
        n_assets = returns.shape[1]

        if verbose:
            print(f"Data: {len(returns)} days, {n_assets} assets")
            print(f"Date range: {returns.index[0].date()} to {returns.index[-1].date()}")

        X_all, y_all, dates_all = prepare_sequences(returns, self.lookback)

        seq_returns = returns.iloc[self.lookback:]
        assert len(seq_returns) == len(X_all)

        validator = WalkForwardValidator(self.config)

        seq_df = pd.DataFrame(index=dates_all, data={"dummy": 0})

        model_results = []
        baseline_results = []
        model_all_returns = []
        baseline_all_returns = []

        window_idx = 0
        for train_idx, test_idx in validator.split(seq_df):
            window_idx += 1
            if verbose:
                print(f"\n--- Window {window_idx}/{self.n_splits} ---")

            X_train = X_all[train_idx]
            y_train = y_all[train_idx]
            X_test = X_all[test_idx]
            y_test = y_all[test_idx]
            test_dates = dates_all[test_idx]

            train_dates = dates_all[train_idx]
            if verbose:
                print(f"  Train: {train_dates[0].date()} to {train_dates[-1].date()} ({len(train_idx)} samples)")
                print(f"  Test:  {test_dates[0].date()} to {test_dates[-1].date()} ({len(test_idx)} samples)")

            # Normalize features using train data only (no leakage)
            scaler = StandardScaler()
            n_samples_train, lookback_dim, n_feat = X_train.shape
            X_train_flat = X_train.reshape(-1, n_feat)
            scaler.fit(X_train_flat)
            X_train_scaled = scaler.transform(X_train_flat).reshape(n_samples_train, lookback_dim, n_feat)

            n_samples_test = X_test.shape[0]
            X_test_flat = X_test.reshape(-1, n_feat)
            X_test_scaled = scaler.transform(X_test_flat).reshape(n_samples_test, lookback_dim, n_feat)

            # Train model
            model = train_model(
                X_train_scaled, y_train, n_assets,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                lr=self.lr,
                epochs=self.epochs,
                batch_size=self.batch_size,
            )

            # Predict weights on test set
            model_weights = predict_weights(model, X_test_scaled)

            # Model portfolio returns
            model_port_returns = (model_weights * y_test).sum(axis=1)
            model_port_series = pd.Series(model_port_returns, index=test_dates)

            # 1/N baseline
            equal_weights = np.ones((len(test_idx), n_assets)) / n_assets
            baseline_port_returns = (equal_weights * y_test).sum(axis=1)
            baseline_port_series = pd.Series(baseline_port_returns, index=test_dates)

            # Compute costs
            model_positions = pd.Series(
                np.ones(len(test_idx)), index=test_dates
            )
            # Approximate trades as weight changes between consecutive days
            weight_changes = np.abs(np.diff(model_weights, axis=0)).sum(axis=1)
            trade_series = pd.Series(
                np.concatenate([[0.0], weight_changes]), index=test_dates
            )

            cost_per_trade = (self.config.fee_bps + self.config.slippage_bps) / 10000
            model_costs = trade_series * cost_per_trade
            model_net_returns = model_port_series - model_costs

            # Baseline has no rebalancing cost (equal weight is maintained by market moves,
            # but we approximate small rebalancing costs)
            baseline_trade_series = pd.Series(0.0, index=test_dates)
            baseline_net_returns = baseline_port_series

            # Compute metrics
            model_gross_metrics = compute_metrics(model_port_series)
            model_net_metrics = compute_metrics(model_net_returns)
            baseline_metrics = compute_metrics(baseline_port_series)

            total_trades = int((trade_series > 0.01).sum())

            model_result = BacktestResult(
                window=window_idx,
                train_start=str(train_dates[0].date()),
                train_end=str(train_dates[-1].date()),
                test_start=str(test_dates[0].date()),
                test_end=str(test_dates[-1].date()),
                gross_sharpe=model_gross_metrics["sharpeRatio"],
                net_sharpe=model_net_metrics["sharpeRatio"],
                annual_return=model_net_metrics["annualReturn"],
                max_drawdown=model_net_metrics["maxDrawdown"],
                total_trades=total_trades,
                hit_rate=model_net_metrics["hitRate"],
                pnl_series=model_net_returns,
            )
            model_results.append(model_result)
            model_all_returns.append(model_net_returns)

            baseline_result = BacktestResult(
                window=window_idx,
                train_start=str(train_dates[0].date()),
                train_end=str(train_dates[-1].date()),
                test_start=str(test_dates[0].date()),
                test_end=str(test_dates[-1].date()),
                gross_sharpe=baseline_metrics["sharpeRatio"],
                net_sharpe=baseline_metrics["sharpeRatio"],
                annual_return=baseline_metrics["annualReturn"],
                max_drawdown=baseline_metrics["maxDrawdown"],
                total_trades=0,
                hit_rate=baseline_metrics["hitRate"],
            )
            baseline_results.append(baseline_result)
            baseline_all_returns.append(baseline_port_series)

            if verbose:
                print(f"  Model  — Gross Sharpe: {model_gross_metrics['sharpeRatio']:.4f}, "
                      f"Net Sharpe: {model_net_metrics['sharpeRatio']:.4f}")
                print(f"  1/N    — Sharpe: {baseline_metrics['sharpeRatio']:.4f}")

        # Aggregate
        model_metrics_json = generate_metrics_json(model_results, self.config)
        baseline_metrics_json = generate_metrics_json(baseline_results, self.config)

        return {
            "model": {
                "results": model_results,
                "metrics": model_metrics_json,
            },
            "baseline": {
                "results": baseline_results,
                "metrics": baseline_metrics_json,
            },
        }


def save_backtest_results(output: dict, output_path: str | Path) -> None:
    """Save backtest results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "model": {
            "metrics": output["model"]["metrics"],
            "windows": [
                {
                    "window": r.window,
                    "train_start": r.train_start,
                    "train_end": r.train_end,
                    "test_start": r.test_start,
                    "test_end": r.test_end,
                    "gross_sharpe": r.gross_sharpe,
                    "net_sharpe": r.net_sharpe,
                    "annual_return": r.annual_return,
                    "max_drawdown": r.max_drawdown,
                    "total_trades": r.total_trades,
                    "hit_rate": r.hit_rate,
                }
                for r in output["model"]["results"]
            ],
        },
        "baseline": {
            "metrics": output["baseline"]["metrics"],
            "windows": [
                {
                    "window": r.window,
                    "train_start": r.train_start,
                    "train_end": r.train_end,
                    "test_start": r.test_start,
                    "test_end": r.test_end,
                    "gross_sharpe": r.gross_sharpe,
                    "net_sharpe": r.net_sharpe,
                    "annual_return": r.annual_return,
                    "max_drawdown": r.max_drawdown,
                    "total_trades": r.total_trades,
                    "hit_rate": r.hit_rate,
                }
                for r in output["baseline"]["results"]
            ],
        },
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

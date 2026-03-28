"""
Walk-forward evaluation framework for portfolio optimization.
Phase 3: Walk-forward validation and baseline comparisons.
Phase 4: Enhanced transaction cost model with sensitivity analysis.
Phase 5: Hyperparameter optimization with Optuna.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from src.backtest import (
    BacktestConfig,
    BacktestResult,
    CostBreakdown,
    TransactionCostModel,
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

    def _run_windows(self, verbose: bool = True) -> dict:
        """Run walk-forward windows and return raw window data.

        Returns dict with per-window gross returns, weights, and metadata.
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

        windows = []
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

            # Model gross portfolio returns
            model_port_returns = (model_weights * y_test).sum(axis=1)
            model_port_series = pd.Series(model_port_returns, index=test_dates)

            # 1/N baseline
            equal_weights = np.ones((len(test_idx), n_assets)) / n_assets
            baseline_port_returns = (equal_weights * y_test).sum(axis=1)
            baseline_port_series = pd.Series(baseline_port_returns, index=test_dates)

            windows.append({
                "window_idx": window_idx,
                "train_dates": train_dates,
                "test_dates": test_dates,
                "model_weights": model_weights,
                "model_gross_returns": model_port_series,
                "baseline_weights": equal_weights,
                "baseline_gross_returns": baseline_port_series,
                "n_assets": n_assets,
            })

        return {"windows": windows, "n_assets": n_assets}

    def _apply_costs_to_windows(
        self,
        window_data: dict,
        cost_model: TransactionCostModel,
        verbose: bool = True,
    ) -> dict:
        """Apply a cost model to window data, returning model and baseline results."""
        model_results = []
        baseline_results = []

        for w in window_data["windows"]:
            # Model costs
            model_net, model_cb = cost_model.apply_costs(
                w["model_gross_returns"], w["model_weights"]
            )
            # Baseline: equal-weight has near-zero turnover (no active rebalancing)
            baseline_net, baseline_cb = cost_model.apply_costs(
                w["baseline_gross_returns"], w["baseline_weights"]
            )

            model_gross_metrics = compute_metrics(w["model_gross_returns"])
            model_net_metrics = compute_metrics(model_net)
            baseline_gross_metrics = compute_metrics(w["baseline_gross_returns"])
            baseline_net_metrics = compute_metrics(baseline_net)

            total_trades = model_cb.n_rebalance_days

            model_result = BacktestResult(
                window=w["window_idx"],
                train_start=str(w["train_dates"][0].date()),
                train_end=str(w["train_dates"][-1].date()),
                test_start=str(w["test_dates"][0].date()),
                test_end=str(w["test_dates"][-1].date()),
                gross_sharpe=model_gross_metrics["sharpeRatio"],
                net_sharpe=model_net_metrics["sharpeRatio"],
                gross_annual_return=model_gross_metrics["annualReturn"],
                annual_return=model_net_metrics["annualReturn"],
                max_drawdown=model_net_metrics["maxDrawdown"],
                total_trades=total_trades,
                hit_rate=model_net_metrics["hitRate"],
                cost_breakdown=model_cb,
                pnl_series=model_net,
            )
            model_results.append(model_result)

            baseline_result = BacktestResult(
                window=w["window_idx"],
                train_start=str(w["train_dates"][0].date()),
                train_end=str(w["train_dates"][-1].date()),
                test_start=str(w["test_dates"][0].date()),
                test_end=str(w["test_dates"][-1].date()),
                gross_sharpe=baseline_gross_metrics["sharpeRatio"],
                net_sharpe=baseline_net_metrics["sharpeRatio"],
                gross_annual_return=baseline_gross_metrics["annualReturn"],
                annual_return=baseline_net_metrics["annualReturn"],
                max_drawdown=baseline_net_metrics["maxDrawdown"],
                total_trades=baseline_cb.n_rebalance_days,
                hit_rate=baseline_net_metrics["hitRate"],
                cost_breakdown=baseline_cb,
            )
            baseline_results.append(baseline_result)

            if verbose:
                print(f"  Window {w['window_idx']} — "
                      f"Model Gross: {model_gross_metrics['sharpeRatio']:.4f}, "
                      f"Net: {model_net_metrics['sharpeRatio']:.4f} | "
                      f"Turnover: {model_cb.avg_daily_turnover:.4f}")

        return {
            "model_results": model_results,
            "baseline_results": baseline_results,
        }

    def run(self, verbose: bool = True) -> dict:
        """Run the full walk-forward backtest.

        Returns:
            Dict with 'model' and 'baseline' results and metrics.
        """
        window_data = self._run_windows(verbose=verbose)
        cost_model = TransactionCostModel.from_config(self.config)

        if verbose:
            print(f"\nApplying costs: {cost_model.fee_bps}bps fee + {cost_model.slippage_bps}bps slippage")

        results = self._apply_costs_to_windows(window_data, cost_model, verbose=verbose)

        model_metrics_json = generate_metrics_json(results["model_results"], self.config)
        baseline_metrics_json = generate_metrics_json(results["baseline_results"], self.config)

        return {
            "model": {
                "results": results["model_results"],
                "metrics": model_metrics_json,
            },
            "baseline": {
                "results": results["baseline_results"],
                "metrics": baseline_metrics_json,
            },
            "_window_data": window_data,
        }

    def run_cost_sensitivity(
        self,
        cost_levels_bps: list[float] | None = None,
        verbose: bool = True,
    ) -> dict:
        """Run cost sensitivity analysis at multiple cost levels.

        Args:
            cost_levels_bps: Total cost levels (fee+slippage) in bps to test.
            verbose: Print progress.

        Returns:
            Dict with sensitivity results per cost level.
        """
        if cost_levels_bps is None:
            cost_levels_bps = [0, 5, 10, 15, 20, 30]

        window_data = self._run_windows(verbose=verbose)

        sensitivity = []
        for total_bps in cost_levels_bps:
            # Split total cost 2:1 between fees and slippage
            fee_bps = round(total_bps * 2 / 3, 1)
            slippage_bps = round(total_bps - fee_bps, 1)
            cost_model = TransactionCostModel(fee_bps=fee_bps, slippage_bps=slippage_bps)

            if verbose:
                print(f"\n--- Cost Level: {total_bps}bps (fee={fee_bps}, slippage={slippage_bps}) ---")

            results = self._apply_costs_to_windows(window_data, cost_model, verbose=verbose)

            config_at_level = BacktestConfig(
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                n_splits=self.config.n_splits,
                gap=self.config.gap,
                min_train_size=self.config.min_train_size,
                train_ratio=self.config.train_ratio,
            )
            model_metrics = generate_metrics_json(results["model_results"], config_at_level)
            baseline_metrics = generate_metrics_json(results["baseline_results"], config_at_level)

            avg_turnover = float(np.mean([
                r.cost_breakdown.avg_daily_turnover
                for r in results["model_results"]
                if r.cost_breakdown
            ]))

            sensitivity.append({
                "total_cost_bps": total_bps,
                "fee_bps": fee_bps,
                "slippage_bps": slippage_bps,
                "model_net_sharpe": model_metrics["transactionCosts"]["netSharpe"],
                "model_gross_sharpe": model_metrics["sharpeRatio"],
                "model_net_annual_return": model_metrics["annualReturn"],
                "baseline_net_sharpe": baseline_metrics["transactionCosts"]["netSharpe"],
                "avg_daily_turnover": round(avg_turnover, 6),
                "positive_windows": model_metrics["walkForward"]["positiveWindows"],
            })

        return {
            "sensitivity": sensitivity,
            "_window_data": window_data,
        }


class OptunaOptimizer:
    """Hyperparameter optimization using Optuna with walk-forward net Sharpe objective."""

    def __init__(
        self,
        tickers: list[str] | None = None,
        n_splits: int = 5,
        fee_bps: float = 10.0,
        slippage_bps: float = 5.0,
        n_trials: int = 20,
        seed: int = 42,
    ):
        self.tickers = tickers
        self.n_splits = n_splits
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        self.n_trials = n_trials
        self.seed = seed
        self._prices = None
        self._returns = None

    def _prefetch_data(self):
        """Fetch data once and cache for all trials."""
        if self._prices is None:
            print("Prefetching data for optimization...")
            self._prices = fetch_universe(self.tickers)
            self._returns = compute_returns(self._prices)
            print(f"Data: {len(self._returns)} days, {self._returns.shape[1]} assets")

    def _objective(self, trial) -> float:
        """Optuna objective: maximize average net Sharpe across walk-forward windows."""
        # Sample hyperparameters (near paper defaults)
        lookback = trial.suggest_categorical("lookback", [30, 45, 60, 90, 120])
        lr = trial.suggest_categorical("lr", [5e-5, 1e-4, 5e-4, 1e-3])
        hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
        num_layers = trial.suggest_int("num_layers", 1, 3)
        epochs = trial.suggest_categorical("epochs", [30, 50, 100])
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

        returns = self._returns
        n_assets = returns.shape[1]

        X_all, y_all, dates_all = prepare_sequences(returns, lookback)
        seq_returns = returns.iloc[lookback:]
        assert len(seq_returns) == len(X_all)

        config = BacktestConfig(
            fee_bps=self.fee_bps,
            slippage_bps=self.slippage_bps,
            n_splits=self.n_splits,
            gap=1,
            min_train_size=max(252, lookback + 60),
            train_ratio=1.0,
        )

        validator = WalkForwardValidator(config)
        seq_df = pd.DataFrame(index=dates_all, data={"dummy": 0})
        cost_model = TransactionCostModel.from_config(config)

        net_sharpes = []
        window_idx = 0
        for train_idx, test_idx in validator.split(seq_df):
            window_idx += 1

            X_train = X_all[train_idx]
            y_train = y_all[train_idx]
            X_test = X_all[test_idx]
            y_test = y_all[test_idx]
            test_dates = dates_all[test_idx]

            # Normalize using train only
            scaler = StandardScaler()
            n_samples_train, lookback_dim, n_feat = X_train.shape
            X_train_flat = X_train.reshape(-1, n_feat)
            scaler.fit(X_train_flat)
            X_train_scaled = scaler.transform(X_train_flat).reshape(n_samples_train, lookback_dim, n_feat)

            n_samples_test = X_test.shape[0]
            X_test_flat = X_test.reshape(-1, n_feat)
            X_test_scaled = scaler.transform(X_test_flat).reshape(n_samples_test, lookback_dim, n_feat)

            from src.model import train_model as _train, predict_weights as _predict
            model = _train(
                X_train_scaled, y_train, n_assets,
                hidden_size=hidden_size,
                num_layers=num_layers,
                lr=lr,
                epochs=epochs,
                batch_size=batch_size,
            )

            weights = _predict(model, X_test_scaled)
            gross_returns = (weights * y_test).sum(axis=1)
            gross_series = pd.Series(gross_returns, index=test_dates)

            net_returns, _ = cost_model.apply_costs(gross_series, weights)
            net_metrics = compute_metrics(net_returns)
            net_sharpes.append(net_metrics["sharpeRatio"])

            # Pruning: report intermediate value
            trial.report(np.mean(net_sharpes), window_idx - 1)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

        avg_net_sharpe = float(np.mean(net_sharpes))
        return avg_net_sharpe

    def optimize(self, verbose: bool = True) -> dict:
        """Run Optuna optimization.

        Returns:
            Dict with best_params, best_value, all_trials, and study.
        """
        import optuna

        self._prefetch_data()

        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name="portfolio_optim",
        )

        if verbose:
            print(f"\nStarting Optuna optimization ({self.n_trials} trials)...")
            print(f"Objective: maximize avg net Sharpe across {self.n_splits} WF windows")
            print(f"Cost model: {self.fee_bps}bps fee + {self.slippage_bps}bps slippage\n")

        study.optimize(self._objective, n_trials=self.n_trials, show_progress_bar=verbose)

        best = study.best_trial
        if verbose:
            print(f"\nBest trial #{best.number}: avg net Sharpe = {best.value:.4f}")
            print(f"Best params: {best.params}")

        trials_data = []
        for t in study.trials:
            trials_data.append({
                "number": t.number,
                "value": t.value if t.value is not None else None,
                "params": t.params,
                "state": str(t.state),
            })

        return {
            "best_params": best.params,
            "best_value": best.value,
            "all_trials": trials_data,
            "study": study,
        }

    def run_best(self, best_params: dict, verbose: bool = True) -> dict:
        """Run full backtest with best parameters and return complete results."""
        evaluator = WalkForwardEvaluator(
            tickers=self.tickers,
            lookback=best_params["lookback"],
            n_splits=self.n_splits,
            hidden_size=best_params["hidden_size"],
            num_layers=best_params["num_layers"],
            lr=best_params["lr"],
            epochs=best_params["epochs"],
            batch_size=best_params["batch_size"],
            fee_bps=self.fee_bps,
            slippage_bps=self.slippage_bps,
        )
        return evaluator.run(verbose=verbose)


def _serialize_result(r: BacktestResult) -> dict:
    """Serialize a BacktestResult to a JSON-compatible dict."""
    d = {
        "window": r.window,
        "train_start": r.train_start,
        "train_end": r.train_end,
        "test_start": r.test_start,
        "test_end": r.test_end,
        "gross_sharpe": r.gross_sharpe,
        "net_sharpe": r.net_sharpe,
        "gross_annual_return": r.gross_annual_return,
        "annual_return": r.annual_return,
        "max_drawdown": r.max_drawdown,
        "total_trades": r.total_trades,
        "hit_rate": r.hit_rate,
    }
    if r.cost_breakdown:
        d["cost_breakdown"] = {
            "total_cost": r.cost_breakdown.total_cost,
            "fee_cost": r.cost_breakdown.fee_cost,
            "slippage_cost": r.cost_breakdown.slippage_cost,
            "total_turnover": r.cost_breakdown.total_turnover,
            "avg_daily_turnover": r.cost_breakdown.avg_daily_turnover,
            "n_rebalance_days": r.cost_breakdown.n_rebalance_days,
        }
    return d


def save_backtest_results(output: dict, output_path: str | Path) -> None:
    """Save backtest results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "model": {
            "metrics": output["model"]["metrics"],
            "windows": [_serialize_result(r) for r in output["model"]["results"]],
        },
        "baseline": {
            "metrics": output["baseline"]["metrics"],
            "windows": [_serialize_result(r) for r in output["baseline"]["results"]],
        },
    }

    # Include cost sensitivity if present
    if "sensitivity" in output:
        result["cost_sensitivity"] = output["sensitivity"]

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

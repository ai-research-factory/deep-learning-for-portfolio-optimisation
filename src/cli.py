"""
CLI for Deep Learning Portfolio Optimization.
"""
import json
import click
from pathlib import Path

from src.evaluation import WalkForwardEvaluator, OptunaOptimizer, save_backtest_results
from src.backtest import generate_metrics_json


@click.group()
def cli():
    """Deep Learning Portfolio Optimization CLI."""
    pass


@cli.command("run-backtest")
@click.option("--config", "config_path", default="configs/default.yaml", help="Config file path")
@click.option("--output-dir", default="reports/cycle_4", help="Output directory for results")
@click.option("--n-splits", default=5, help="Number of walk-forward splits")
@click.option("--lookback", default=60, help="Lookback window in days")
@click.option("--epochs", default=50, help="Training epochs per window")
@click.option("--lr", default=1e-4, type=float, help="Learning rate")
@click.option("--hidden-size", default=64, help="LSTM hidden size")
@click.option("--fee-bps", default=10.0, type=float, help="Transaction fee in bps")
@click.option("--slippage-bps", default=5.0, type=float, help="Slippage in bps")
def run_backtest(config_path, output_dir, n_splits, lookback, epochs, lr, hidden_size, fee_bps, slippage_bps):
    """Run walk-forward backtest with LSTM model vs 1/N baseline."""
    import yaml

    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    tickers = cfg.get("tickers", None)
    n_splits = cfg.get("n_splits", n_splits)
    lookback = cfg.get("lookback", lookback)
    epochs = cfg.get("epochs", epochs)
    lr = cfg.get("lr", lr)
    hidden_size = cfg.get("hidden_size", hidden_size)
    fee_bps = cfg.get("fee_bps", fee_bps)
    slippage_bps = cfg.get("slippage_bps", slippage_bps)

    print(f"Walk-Forward Backtest Configuration:")
    print(f"  n_splits={n_splits}, lookback={lookback}, epochs={epochs}, lr={lr}")
    print(f"  fee_bps={fee_bps}, slippage_bps={slippage_bps}")

    evaluator = WalkForwardEvaluator(
        tickers=tickers,
        lookback=lookback,
        n_splits=n_splits,
        hidden_size=hidden_size,
        lr=lr,
        epochs=epochs,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )

    output = evaluator.run(verbose=True)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save detailed backtest results
    backtest_path = output_path / "backtest_results.json"
    save_backtest_results(output, backtest_path)
    print(f"\nBacktest results saved to {backtest_path}")

    # Save metrics.json (model metrics as primary)
    metrics = output["model"]["metrics"]
    metrics["customMetrics"] = {
        "baseline_sharpe": output["baseline"]["metrics"]["sharpeRatio"],
        "baseline_net_sharpe": output["baseline"]["metrics"]["transactionCosts"]["netSharpe"],
        "baseline_annual_return": output["baseline"]["metrics"]["annualReturn"],
        "baseline_max_drawdown": output["baseline"]["metrics"]["maxDrawdown"],
    }
    metrics_path = output_path / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    _print_summary(metrics, output["baseline"]["metrics"])


@cli.command("run-cost-analysis")
@click.option("--config", "config_path", default="configs/default.yaml", help="Config file path")
@click.option("--output-dir", default="reports/cycle_4", help="Output directory for results")
@click.option("--cost-levels", default="0,5,10,15,20,30", help="Comma-separated cost levels in bps")
def run_cost_analysis(config_path, output_dir, cost_levels):
    """Run cost sensitivity analysis at multiple cost levels."""
    import yaml

    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    tickers = cfg.get("tickers", None)
    n_splits = cfg.get("n_splits", 5)
    lookback = cfg.get("lookback", 60)
    epochs = cfg.get("epochs", 50)
    lr = cfg.get("lr", 1e-4)
    hidden_size = cfg.get("hidden_size", 64)

    levels = [float(x) for x in cost_levels.split(",")]
    print(f"Cost Sensitivity Analysis")
    print(f"  Cost levels (bps): {levels}")

    evaluator = WalkForwardEvaluator(
        tickers=tickers,
        lookback=lookback,
        n_splits=n_splits,
        hidden_size=hidden_size,
        lr=lr,
        epochs=epochs,
    )

    result = evaluator.run_cost_sensitivity(cost_levels_bps=levels, verbose=True)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sensitivity_path = output_path / "cost_sensitivity.json"
    with open(sensitivity_path, "w") as f:
        json.dump(result["sensitivity"], f, indent=2)
    print(f"\nCost sensitivity saved to {sensitivity_path}")

    # Print table
    print("\n" + "=" * 80)
    print("COST SENSITIVITY RESULTS")
    print("=" * 80)
    print(f"{'Cost (bps)':>10} | {'Model Net Sharpe':>16} | {'Model Gross Sharpe':>18} | {'1/N Net Sharpe':>14} | {'Turnover':>10}")
    print("-" * 80)
    for s in result["sensitivity"]:
        print(f"{s['total_cost_bps']:>10.0f} | {s['model_net_sharpe']:>16.4f} | {s['model_gross_sharpe']:>18.4f} | {s['baseline_net_sharpe']:>14.4f} | {s['avg_daily_turnover']:>10.4f}")


@cli.command("run-optimize")
@click.option("--config", "config_path", default="configs/default.yaml", help="Config file path")
@click.option("--output-dir", default="reports/cycle_5", help="Output directory for results")
@click.option("--n-trials", default=20, help="Number of Optuna trials")
@click.option("--n-splits", default=5, help="Number of walk-forward splits")
@click.option("--fee-bps", default=10.0, type=float, help="Transaction fee in bps")
@click.option("--slippage-bps", default=5.0, type=float, help="Slippage in bps")
@click.option("--seed", default=42, help="Random seed for reproducibility")
def run_optimize(config_path, output_dir, n_trials, n_splits, fee_bps, slippage_bps, seed):
    """Run Optuna hyperparameter optimization for LSTM portfolio model."""
    import yaml

    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    tickers = cfg.get("tickers", None)
    n_splits = cfg.get("n_splits", n_splits)
    fee_bps = cfg.get("fee_bps", fee_bps)
    slippage_bps = cfg.get("slippage_bps", slippage_bps)

    print("=" * 60)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    print(f"  n_trials={n_trials}, n_splits={n_splits}, seed={seed}")
    print(f"  fee_bps={fee_bps}, slippage_bps={slippage_bps}")

    optimizer = OptunaOptimizer(
        tickers=tickers,
        n_splits=n_splits,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        n_trials=n_trials,
        seed=seed,
    )

    optuna_result = optimizer.optimize(verbose=True)

    # Run full backtest with best params
    print("\n" + "=" * 60)
    print("RUNNING FULL BACKTEST WITH BEST PARAMETERS")
    print("=" * 60)
    best_output = optimizer.run_best(optuna_result["best_params"], verbose=True)

    # Also run baseline (default params) for comparison
    print("\n" + "=" * 60)
    print("RUNNING BASELINE BACKTEST (DEFAULT PARAMETERS)")
    print("=" * 60)
    default_evaluator = WalkForwardEvaluator(
        tickers=tickers,
        lookback=60,
        n_splits=n_splits,
        hidden_size=64,
        num_layers=2,
        lr=1e-4,
        epochs=50,
        batch_size=64,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )
    default_output = default_evaluator.run(verbose=True)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save detailed backtest results (optimized)
    backtest_path = output_path / "backtest_results.json"
    save_backtest_results(best_output, backtest_path)
    print(f"\nOptimized backtest results saved to {backtest_path}")

    # Build metrics.json
    opt_metrics = best_output["model"]["metrics"]
    default_metrics = default_output["model"]["metrics"]
    baseline_1n = best_output["baseline"]["metrics"]

    # Collect per-window turnover from optimized results
    per_window_turnover = {}
    for r in best_output["model"]["results"]:
        if r.cost_breakdown:
            per_window_turnover[f"window_{r.window}"] = round(r.cost_breakdown.avg_daily_turnover, 4)

    opt_metrics["customMetrics"] = {
        "baseline_sharpe": baseline_1n["sharpeRatio"],
        "baseline_net_sharpe": baseline_1n["transactionCosts"]["netSharpe"],
        "baseline_annual_return": baseline_1n["annualReturn"],
        "baseline_max_drawdown": baseline_1n["maxDrawdown"],
        "default_params_net_sharpe": default_metrics["transactionCosts"]["netSharpe"],
        "default_params_gross_sharpe": default_metrics["sharpeRatio"],
        "optimization": {
            "n_trials": n_trials,
            "best_trial": optuna_result["study"].best_trial.number,
            "best_params": optuna_result["best_params"],
            "best_avg_net_sharpe": optuna_result["best_value"],
            "improvement_over_default": round(
                opt_metrics["transactionCosts"]["netSharpe"]
                - default_metrics["transactionCosts"]["netSharpe"], 4
            ),
        },
        "per_window_turnover": per_window_turnover,
    }

    metrics_path = output_path / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(opt_metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Save optimization history
    optuna_path = output_path / "optuna_trials.json"
    with open(optuna_path, "w") as f:
        json.dump(optuna_result["all_trials"], f, indent=2)
    print(f"Optuna trials saved to {optuna_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nBest Parameters:")
    for k, v in optuna_result["best_params"].items():
        print(f"  {k}: {v}")
    print(f"\nOptimized Model:")
    print(f"  Gross Sharpe:    {opt_metrics['sharpeRatio']:.4f}")
    print(f"  Net Sharpe:      {opt_metrics['transactionCosts']['netSharpe']:.4f}")
    print(f"  Annual Return:   {opt_metrics['annualReturn']:.4f}")
    print(f"  Max Drawdown:    {opt_metrics['maxDrawdown']:.4f}")
    print(f"  Positive Windows: {opt_metrics['walkForward']['positiveWindows']}/{opt_metrics['walkForward']['windows']}")
    print(f"\nDefault Parameters Model:")
    print(f"  Net Sharpe:      {default_metrics['transactionCosts']['netSharpe']:.4f}")
    print(f"\n1/N Baseline:")
    print(f"  Net Sharpe:      {baseline_1n['transactionCosts']['netSharpe']:.4f}")
    print(f"\nImprovement (optimized vs default): {opt_metrics['transactionCosts']['netSharpe'] - default_metrics['transactionCosts']['netSharpe']:.4f} Sharpe")


@cli.command("run-experiment")
@click.option("--config", "config_path", default="configs/default.yaml", help="Config file path")
def run_experiment(config_path):
    """Run experiment (alias for run-backtest)."""
    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(run_backtest, ["--config", config_path])
    print(result.output)


def _print_summary(metrics: dict, baseline_metrics: dict):
    """Print results summary."""
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Model  — Gross Sharpe:    {metrics['sharpeRatio']:.4f}")
    print(f"         Net Sharpe:      {metrics['transactionCosts']['netSharpe']:.4f}")
    print(f"         Annual Return:   {metrics['annualReturn']:.4f}")
    print(f"         Max Drawdown:    {metrics['maxDrawdown']:.4f}")
    print(f"         Positive Windows: {metrics['walkForward']['positiveWindows']}/{metrics['walkForward']['windows']}")
    print(f"         Cost Impact:     {metrics['sharpeRatio'] - metrics['transactionCosts']['netSharpe']:.4f} Sharpe")
    print(f"1/N    — Gross Sharpe:    {baseline_metrics['sharpeRatio']:.4f}")
    print(f"         Net Sharpe:      {baseline_metrics['transactionCosts']['netSharpe']:.4f}")
    print(f"         Annual Return:   {baseline_metrics['annualReturn']:.4f}")
    print(f"         Max Drawdown:    {baseline_metrics['maxDrawdown']:.4f}")


if __name__ == "__main__":
    cli()

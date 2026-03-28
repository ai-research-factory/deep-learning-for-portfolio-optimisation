"""
CLI for Deep Learning Portfolio Optimization.
"""
import json
import click
from pathlib import Path

from src.evaluation import WalkForwardEvaluator, save_backtest_results
from src.backtest import generate_metrics_json


@click.group()
def cli():
    """Deep Learning Portfolio Optimization CLI."""
    pass


@cli.command("run-backtest")
@click.option("--config", "config_path", default="configs/default.yaml", help="Config file path")
@click.option("--output-dir", default="reports/cycle_3", help="Output directory for results")
@click.option("--n-splits", default=5, help="Number of walk-forward splits")
@click.option("--lookback", default=60, help="Lookback window in days")
@click.option("--epochs", default=50, help="Training epochs per window")
@click.option("--lr", default=1e-4, type=float, help="Learning rate")
@click.option("--hidden-size", default=64, help="LSTM hidden size")
def run_backtest(config_path, output_dir, n_splits, lookback, epochs, lr, hidden_size):
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

    print(f"Walk-Forward Backtest Configuration:")
    print(f"  n_splits={n_splits}, lookback={lookback}, epochs={epochs}, lr={lr}")

    evaluator = WalkForwardEvaluator(
        tickers=tickers,
        lookback=lookback,
        n_splits=n_splits,
        hidden_size=hidden_size,
        lr=lr,
        epochs=epochs,
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
        "baseline_annual_return": output["baseline"]["metrics"]["annualReturn"],
        "baseline_max_drawdown": output["baseline"]["metrics"]["maxDrawdown"],
    }
    metrics_path = output_path / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Model  — Avg OOS Sharpe: {metrics['walkForward']['avgOosSharpe']:.4f}")
    print(f"         Annual Return:  {metrics['annualReturn']:.4f}")
    print(f"         Max Drawdown:   {metrics['maxDrawdown']:.4f}")
    print(f"         Positive Windows: {metrics['walkForward']['positiveWindows']}/{metrics['walkForward']['windows']}")
    bm = output["baseline"]["metrics"]
    print(f"1/N    — Avg OOS Sharpe: {bm['walkForward']['avgOosSharpe']:.4f}")
    print(f"         Annual Return:  {bm['annualReturn']:.4f}")
    print(f"         Max Drawdown:   {bm['maxDrawdown']:.4f}")


@cli.command("run-experiment")
@click.option("--config", "config_path", default="configs/default.yaml", help="Config file path")
def run_experiment(config_path):
    """Run experiment (alias for run-backtest)."""
    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(run_backtest, ["--config", config_path])
    print(result.output)


if __name__ == "__main__":
    cli()

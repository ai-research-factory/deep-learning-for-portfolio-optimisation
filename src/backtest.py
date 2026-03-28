"""
ARF Standard Backtest Framework
Walk-forward validation with transaction cost accounting.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    fee_bps: float = 10.0       # Transaction fee in basis points
    slippage_bps: float = 5.0   # Slippage in basis points
    train_ratio: float = 0.7    # Train window ratio for walk-forward
    n_splits: int = 10          # Number of walk-forward windows
    gap: int = 1                # Gap between train and test (prevent leakage)
    min_train_size: int = 252   # Minimum training samples (~1 year daily)


@dataclass
class CostBreakdown:
    """Detailed transaction cost breakdown for a window."""
    total_cost: float = 0.0
    fee_cost: float = 0.0
    slippage_cost: float = 0.0
    total_turnover: float = 0.0
    avg_daily_turnover: float = 0.0
    n_rebalance_days: int = 0


@dataclass
class BacktestResult:
    """Results from a single walk-forward window."""
    window: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    gross_sharpe: float = 0.0
    net_sharpe: float = 0.0
    gross_annual_return: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    hit_rate: float = 0.0
    cost_breakdown: Optional[CostBreakdown] = field(default=None, repr=False)
    pnl_series: Optional[pd.Series] = field(default=None, repr=False)


class WalkForwardValidator:
    """
    Walk-forward out-of-sample validation.

    Usage:
        validator = WalkForwardValidator(config)
        for train_idx, test_idx in validator.split(df):
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]
            # Train model on train_df, evaluate on test_df
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def split(self, data: pd.DataFrame):
        """Generate train/test index pairs for walk-forward validation."""
        n = len(data)
        cfg = self.config
        test_size = max(1, (n - cfg.min_train_size) // cfg.n_splits)

        for i in range(cfg.n_splits):
            test_end = n - (cfg.n_splits - 1 - i) * test_size
            test_start = test_end - test_size
            train_end = test_start - cfg.gap
            train_start = max(0, int(train_end * (1 - cfg.train_ratio))) if cfg.train_ratio < 1.0 else 0

            if train_end - train_start < cfg.min_train_size:
                continue
            if test_start >= test_end:
                continue

            yield (
                list(range(train_start, train_end)),
                list(range(test_start, test_end)),
            )


class TransactionCostModel:
    """Detailed transaction cost model with proportional fees and slippage.

    Computes costs from portfolio weight changes (turnover) each day,
    applying separate fee and slippage components.
    """

    def __init__(self, fee_bps: float = 10.0, slippage_bps: float = 5.0):
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps

    @property
    def total_cost_bps(self) -> float:
        return self.fee_bps + self.slippage_bps

    def compute_turnover(self, weights: np.ndarray) -> np.ndarray:
        """Compute daily turnover from weight changes.

        Args:
            weights: Array of shape (T, n_assets) with portfolio weights.

        Returns:
            Array of shape (T,) with daily turnover (sum of absolute weight changes).
        """
        weight_changes = np.abs(np.diff(weights, axis=0)).sum(axis=1)
        return np.concatenate([[0.0], weight_changes])

    def apply_costs(
        self,
        gross_returns: pd.Series,
        weights: np.ndarray,
    ) -> tuple[pd.Series, CostBreakdown]:
        """Apply transaction costs to gross portfolio returns.

        Args:
            gross_returns: Gross portfolio returns series.
            weights: Portfolio weights array (T, n_assets).

        Returns:
            Tuple of (net_returns, cost_breakdown).
        """
        turnover = self.compute_turnover(weights)
        turnover_series = pd.Series(turnover, index=gross_returns.index)

        fee_costs = turnover_series * (self.fee_bps / 10000)
        slippage_costs = turnover_series * (self.slippage_bps / 10000)
        total_costs = fee_costs + slippage_costs

        net_returns = gross_returns - total_costs

        n_rebalance = int((turnover_series > 0.01).sum())
        breakdown = CostBreakdown(
            total_cost=float(total_costs.sum()),
            fee_cost=float(fee_costs.sum()),
            slippage_cost=float(slippage_costs.sum()),
            total_turnover=float(turnover_series.sum()),
            avg_daily_turnover=float(turnover_series.mean()),
            n_rebalance_days=n_rebalance,
        )

        return net_returns, breakdown

    @classmethod
    def from_config(cls, config: BacktestConfig) -> "TransactionCostModel":
        return cls(fee_bps=config.fee_bps, slippage_bps=config.slippage_bps)


def calculate_costs(returns: pd.Series, positions: pd.Series, config: BacktestConfig) -> pd.Series:
    """
    Calculate transaction costs from position changes.

    Args:
        returns: Gross returns series
        positions: Position series (-1, 0, 1 or continuous)
        config: Backtest configuration with fee/slippage settings

    Returns:
        Net returns after costs
    """
    trades = positions.diff().abs().fillna(0)
    cost_per_trade = (config.fee_bps + config.slippage_bps) / 10000
    costs = trades * cost_per_trade
    return returns - costs


def compute_metrics(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> dict:
    """
    Compute standard performance metrics from a returns series.

    Args:
        returns: Daily (or periodic) returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year (252 for daily, 365 for crypto)

    Returns:
        Dict with sharpeRatio, annualReturn, maxDrawdown, hitRate, totalTrades
    """
    if len(returns) == 0:
        return {"sharpeRatio": 0.0, "annualReturn": 0.0, "maxDrawdown": 0.0, "hitRate": 0.0}

    excess = returns - risk_free_rate / periods_per_year
    sharpe = float(np.sqrt(periods_per_year) * excess.mean() / excess.std()) if excess.std() > 0 else 0.0

    cumulative = (1 + returns).cumprod()
    annual_return = float(cumulative.iloc[-1] ** (periods_per_year / len(returns)) - 1)

    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = float(drawdown.min())

    hit_rate = float((returns > 0).sum() / len(returns)) if len(returns) > 0 else 0.0

    return {
        "sharpeRatio": round(sharpe, 4),
        "annualReturn": round(annual_return, 4),
        "maxDrawdown": round(max_drawdown, 4),
        "hitRate": round(hit_rate, 4),
    }


def generate_metrics_json(
    results: list[BacktestResult],
    config: BacktestConfig,
    custom_metrics: Optional[dict] = None,
) -> dict:
    """
    Generate ARF-standard metrics.json from walk-forward results.

    Args:
        results: List of BacktestResult from each window
        config: Backtest configuration
        custom_metrics: Optional paper-specific metrics

    Returns:
        Dict matching ARF metrics.json schema
    """
    if not results:
        return {
            "sharpeRatio": 0.0, "annualReturn": 0.0, "maxDrawdown": 0.0,
            "hitRate": 0.0, "totalTrades": 0,
            "transactionCosts": {"feeBps": config.fee_bps, "slippageBps": config.slippage_bps, "netSharpe": 0.0},
            "walkForward": {"windows": 0, "positiveWindows": 0, "avgOosSharpe": 0.0},
            "customMetrics": custom_metrics or {},
        }

    net_sharpes = [r.net_sharpe for r in results]
    positive_windows = sum(1 for s in net_sharpes if s > 0)

    return {
        "sharpeRatio": round(float(np.mean([r.gross_sharpe for r in results])), 4),
        "annualReturn": round(float(np.mean([r.annual_return for r in results])), 4),
        "maxDrawdown": round(float(min(r.max_drawdown for r in results)), 4),
        "hitRate": round(float(np.mean([r.hit_rate for r in results])), 4),
        "totalTrades": sum(r.total_trades for r in results),
        "transactionCosts": {
            "feeBps": config.fee_bps,
            "slippageBps": config.slippage_bps,
            "netSharpe": round(float(np.mean(net_sharpes)), 4),
        },
        "walkForward": {
            "windows": len(results),
            "positiveWindows": positive_windows,
            "avgOosSharpe": round(float(np.mean(net_sharpes)), 4),
        },
        "customMetrics": custom_metrics or {},
    }

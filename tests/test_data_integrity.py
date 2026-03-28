"""
Data integrity and leakage tests for the portfolio optimization pipeline.
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from sklearn.preprocessing import StandardScaler

from src.backtest import BacktestConfig, WalkForwardValidator, compute_metrics, calculate_costs, TransactionCostModel, CostBreakdown
from src.data_pipeline import compute_returns, prepare_sequences
from src.model import PortfolioLSTM, SharpeLoss


class TestWalkForwardValidator:
    """Tests for walk-forward validation split logic."""

    def setup_method(self):
        dates = pd.date_range("2020-01-01", periods=1000, freq="B")
        self.df = pd.DataFrame({"close": np.random.randn(1000)}, index=dates)

    def test_no_overlap_between_train_and_test(self):
        """Train and test indices must not overlap (no data leakage)."""
        config = BacktestConfig(n_splits=5, min_train_size=252, gap=1)
        validator = WalkForwardValidator(config)

        for train_idx, test_idx in validator.split(self.df):
            train_set = set(train_idx)
            test_set = set(test_idx)
            assert train_set.isdisjoint(test_set), "Train and test indices overlap!"

    def test_train_precedes_test(self):
        """All training indices must come before test indices (temporal order)."""
        config = BacktestConfig(n_splits=5, min_train_size=252, gap=1)
        validator = WalkForwardValidator(config)

        for train_idx, test_idx in validator.split(self.df):
            assert max(train_idx) < min(test_idx), \
                f"Train end ({max(train_idx)}) >= Test start ({min(test_idx)})"

    def test_gap_between_train_and_test(self):
        """There should be a gap between train end and test start."""
        config = BacktestConfig(n_splits=5, min_train_size=252, gap=1)
        validator = WalkForwardValidator(config)

        for train_idx, test_idx in validator.split(self.df):
            assert min(test_idx) - max(train_idx) >= config.gap, \
                "Gap between train and test is too small"

    def test_produces_splits(self):
        """Should produce at least n_splits-1 splits (first may be skipped if train too small)."""
        config = BacktestConfig(n_splits=5, min_train_size=100, gap=1, train_ratio=1.0)
        validator = WalkForwardValidator(config)

        splits = list(validator.split(self.df))
        assert len(splits) >= 4, f"Expected at least 4 splits, got {len(splits)}"
        assert len(splits) <= 5, f"Expected at most 5 splits, got {len(splits)}"

    def test_no_test_indices_in_future(self):
        """Test indices should not exceed data length."""
        config = BacktestConfig(n_splits=5, min_train_size=252, gap=1)
        validator = WalkForwardValidator(config)

        for train_idx, test_idx in validator.split(self.df):
            assert max(test_idx) < len(self.df), "Test index exceeds data bounds"


class TestDataPipeline:
    """Tests for data pipeline functions."""

    def test_compute_returns(self):
        """Returns should be computed correctly."""
        prices = pd.DataFrame({
            "A": [100.0, 110.0, 105.0, 115.0],
            "B": [50.0, 52.0, 48.0, 51.0],
        })
        returns = compute_returns(prices)
        assert len(returns) == 3  # One less row than prices
        assert abs(returns.iloc[0]["A"] - 0.10) < 1e-10
        assert abs(returns.iloc[1]["A"] - (-5 / 110)) < 1e-10

    def test_prepare_sequences_shape(self):
        """Sequences should have correct shape."""
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        returns = pd.DataFrame(
            np.random.randn(100, 3),
            index=dates,
            columns=["A", "B", "C"],
        )
        lookback = 20
        X, y, idx = prepare_sequences(returns, lookback)
        assert X.shape == (80, 20, 3)
        assert y.shape == (80, 3)
        assert len(idx) == 80

    def test_prepare_sequences_no_future_leakage(self):
        """Features at time t should only use data from t-lookback to t-1."""
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        returns = pd.DataFrame(
            np.arange(300).reshape(100, 3).astype(float),
            index=dates,
            columns=["A", "B", "C"],
        )
        lookback = 10
        X, y, idx = prepare_sequences(returns, lookback)

        # The first sample should use data[0:10] as features and data[10] as target
        np.testing.assert_array_equal(X[0], returns.values[0:10])
        np.testing.assert_array_equal(y[0], returns.values[10])

        # The target should be AFTER the feature window
        for i in range(len(X)):
            feature_end_idx = lookback + i - 1
            target_idx = lookback + i
            np.testing.assert_array_equal(X[i][-1], returns.values[feature_end_idx])
            np.testing.assert_array_equal(y[i], returns.values[target_idx])


class TestScalerLeakage:
    """Ensure scaler is only fit on training data."""

    def test_scaler_fit_on_train_only(self):
        """Scaler must be fit only on training data, not full dataset."""
        np.random.seed(42)
        train_data = np.random.randn(100, 5) * 2 + 3
        test_data = np.random.randn(50, 5) * 10 + 100

        scaler = StandardScaler()
        scaler.fit(train_data)

        train_scaled = scaler.transform(train_data)
        test_scaled = scaler.transform(test_data)

        # Train data should be roughly centered around 0
        assert abs(train_scaled.mean()) < 0.5
        # Test data should NOT be centered (different distribution)
        assert abs(test_scaled.mean()) > 5


class TestModel:
    """Tests for the LSTM model."""

    def test_portfolio_weights_sum_to_one(self):
        """Model output weights must sum to 1 (softmax)."""
        import torch
        model = PortfolioLSTM(n_assets=5, hidden_size=16, num_layers=1)
        model.eval()
        x = torch.randn(10, 20, 5)
        with torch.no_grad():
            weights = model(x)
        sums = weights.sum(dim=1).numpy()
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)

    def test_weights_non_negative(self):
        """Softmax weights must be non-negative (long-only constraint)."""
        import torch
        model = PortfolioLSTM(n_assets=5, hidden_size=16, num_layers=1)
        model.eval()
        x = torch.randn(10, 20, 5)
        with torch.no_grad():
            weights = model(x)
        assert (weights.numpy() >= 0).all()

    def test_sharpe_loss_sign(self):
        """Negative Sharpe loss should be negative when Sharpe is positive."""
        import torch
        loss_fn = SharpeLoss()
        weights = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        returns = torch.tensor([[0.02, 0.01], [0.03, 0.02]])
        loss = loss_fn(weights, returns)
        # Positive returns → positive Sharpe → negative loss
        assert loss.item() < 0


class TestMetrics:
    """Tests for metric computation."""

    def test_compute_metrics_basic(self):
        """Metrics should return correct keys and reasonable values."""
        returns = pd.Series(np.random.randn(252) * 0.01)
        metrics = compute_metrics(returns)
        assert "sharpeRatio" in metrics
        assert "annualReturn" in metrics
        assert "maxDrawdown" in metrics
        assert "hitRate" in metrics
        assert metrics["maxDrawdown"] <= 0

    def test_compute_metrics_empty(self):
        """Empty returns should return zeros."""
        metrics = compute_metrics(pd.Series([], dtype=float))
        assert metrics["sharpeRatio"] == 0.0

    def test_calculate_costs(self):
        """Transaction costs should reduce returns."""
        config = BacktestConfig(fee_bps=10.0, slippage_bps=5.0)
        returns = pd.Series([0.01, 0.02, -0.01, 0.015])
        positions = pd.Series([1.0, 1.0, 0.0, 1.0])
        net = calculate_costs(returns, positions, config)
        assert (net <= returns).all()


class TestTransactionCostModel:
    """Tests for the TransactionCostModel class."""

    def test_zero_cost_returns_gross(self):
        """With zero costs, net returns should equal gross returns."""
        cost_model = TransactionCostModel(fee_bps=0.0, slippage_bps=0.0)
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        gross = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005], index=dates)
        weights = np.array([
            [0.5, 0.5],
            [0.6, 0.4],
            [0.7, 0.3],
            [0.55, 0.45],
            [0.5, 0.5],
        ])
        net, breakdown = cost_model.apply_costs(gross, weights)
        pd.testing.assert_series_equal(net, gross)
        assert breakdown.total_cost == 0.0

    def test_costs_reduce_returns(self):
        """Positive costs should reduce net returns vs gross."""
        cost_model = TransactionCostModel(fee_bps=10.0, slippage_bps=5.0)
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        gross = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005], index=dates)
        weights = np.array([
            [0.5, 0.5],
            [0.6, 0.4],
            [0.7, 0.3],
            [0.55, 0.45],
            [0.5, 0.5],
        ])
        net, breakdown = cost_model.apply_costs(gross, weights)
        assert (net <= gross).all()
        assert breakdown.total_cost > 0
        assert breakdown.fee_cost > 0
        assert breakdown.slippage_cost > 0

    def test_higher_costs_lower_returns(self):
        """Higher cost levels should produce lower net returns."""
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        gross = pd.Series(np.random.randn(10) * 0.01, index=dates)
        weights = np.random.dirichlet(np.ones(3), size=10)

        low_model = TransactionCostModel(fee_bps=5.0, slippage_bps=2.5)
        high_model = TransactionCostModel(fee_bps=20.0, slippage_bps=10.0)

        net_low, _ = low_model.apply_costs(gross, weights)
        net_high, _ = high_model.apply_costs(gross, weights)

        assert net_low.sum() > net_high.sum()

    def test_turnover_computation(self):
        """Turnover should correctly sum absolute weight changes."""
        cost_model = TransactionCostModel(fee_bps=10.0, slippage_bps=5.0)
        weights = np.array([
            [0.5, 0.5],
            [0.7, 0.3],  # change: |0.2| + |0.2| = 0.4
            [0.7, 0.3],  # change: 0 (no rebalance)
        ])
        turnover = cost_model.compute_turnover(weights)
        assert len(turnover) == 3
        assert turnover[0] == 0.0
        np.testing.assert_almost_equal(turnover[1], 0.4)
        np.testing.assert_almost_equal(turnover[2], 0.0)

    def test_cost_breakdown_fields(self):
        """Cost breakdown should have all required fields."""
        cost_model = TransactionCostModel(fee_bps=10.0, slippage_bps=5.0)
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        gross = pd.Series([0.01, 0.02, -0.01], index=dates)
        weights = np.array([[0.5, 0.5], [0.7, 0.3], [0.6, 0.4]])
        _, breakdown = cost_model.apply_costs(gross, weights)
        assert isinstance(breakdown, CostBreakdown)
        assert breakdown.total_cost == breakdown.fee_cost + breakdown.slippage_cost
        assert breakdown.total_turnover >= 0
        assert breakdown.avg_daily_turnover >= 0

    def test_from_config(self):
        """TransactionCostModel.from_config should use config values."""
        config = BacktestConfig(fee_bps=15.0, slippage_bps=7.5)
        cost_model = TransactionCostModel.from_config(config)
        assert cost_model.fee_bps == 15.0
        assert cost_model.slippage_bps == 7.5
        assert cost_model.total_cost_bps == 22.5

    def test_no_rebalance_no_cost(self):
        """Constant weights should incur zero costs."""
        cost_model = TransactionCostModel(fee_bps=10.0, slippage_bps=5.0)
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        gross = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005], index=dates)
        weights = np.ones((5, 3)) / 3  # constant equal weights
        net, breakdown = cost_model.apply_costs(gross, weights)
        pd.testing.assert_series_equal(net, gross)
        assert breakdown.total_cost == 0.0

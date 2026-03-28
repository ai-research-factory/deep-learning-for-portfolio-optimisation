"""
Data pipeline for fetching and preprocessing ETF data from the ARF Data API.
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path

API_BASE = "https://ai.1s.xyz/api/data/ohlcv"

DEFAULT_UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "LQD", "GLD", "VNQ", "DBC"]

DATA_DIR = Path(__file__).parent.parent / "data"


def fetch_ohlcv(ticker: str, interval: str = "1d", period: str = "5y", cache: bool = True) -> pd.DataFrame:
    """Fetch OHLCV data from ARF Data API with local caching."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_DIR / f"{ticker}_{interval}_{period}.csv"

    if cache and cache_path.exists():
        df = pd.read_csv(cache_path, parse_dates=["timestamp"], index_col="timestamp")
        return df

    url = f"{API_BASE}?ticker={ticker}&interval={interval}&period={period}"
    df = pd.read_csv(url, parse_dates=["timestamp"], index_col="timestamp")
    df = df.sort_index()

    if cache:
        df.to_csv(cache_path)

    return df


def fetch_universe(
    tickers: list[str] | None = None,
    interval: str = "1d",
    period: str = "5y",
) -> pd.DataFrame:
    """Fetch close prices for all tickers in the universe.

    Returns:
        DataFrame with DatetimeIndex and one column per ticker (close prices).
    """
    tickers = tickers or DEFAULT_UNIVERSE
    closes = {}
    for ticker in tickers:
        df = fetch_ohlcv(ticker, interval=interval, period=period)
        closes[ticker] = df["close"]

    prices = pd.DataFrame(closes)
    prices = prices.dropna()
    prices = prices.sort_index()
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple daily returns from close prices."""
    return prices.pct_change().dropna()


def prepare_sequences(
    returns: pd.DataFrame,
    lookback: int = 60,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Create rolling window sequences for the LSTM model.

    Args:
        returns: DataFrame of daily returns (T x N_assets).
        lookback: Number of past days to use as features.

    Returns:
        X: array of shape (n_samples, lookback, n_assets)
        y: array of shape (n_samples, n_assets) — next-day returns
        dates: DatetimeIndex of the prediction dates
    """
    data = returns.values
    dates = returns.index
    X, y, idx = [], [], []

    for t in range(lookback, len(data)):
        X.append(data[t - lookback:t])
        y.append(data[t])
        idx.append(dates[t])

    return np.array(X), np.array(y), pd.DatetimeIndex(idx)

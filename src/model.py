"""
LSTM portfolio optimization model with Sharpe ratio loss.
"""
import torch
import torch.nn as nn
import numpy as np


class SharpeLoss(nn.Module):
    """Negative Sharpe ratio loss for portfolio optimization."""

    def forward(self, weights: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """
        Args:
            weights: Portfolio weights (batch_size, n_assets) — output of model
            returns: Actual next-day returns (batch_size, n_assets)

        Returns:
            Negative Sharpe ratio (scalar) for the batch.
        """
        portfolio_returns = (weights * returns).sum(dim=1)
        mean_ret = portfolio_returns.mean()
        std_ret = portfolio_returns.std()

        if std_ret < 1e-8:
            return -mean_ret

        sharpe = mean_ret / std_ret
        return -sharpe


class PortfolioLSTM(nn.Module):
    """LSTM model that outputs portfolio weights via softmax."""

    def __init__(self, n_assets: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.n_assets = n_assets
        self.lstm = nn.LSTM(
            input_size=n_assets,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, n_assets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, lookback, n_assets)

        Returns:
            Portfolio weights of shape (batch_size, n_assets) summing to 1.
        """
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        logits = self.fc(last_hidden)
        weights = torch.softmax(logits, dim=1)
        return weights


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_assets: int,
    hidden_size: int = 64,
    num_layers: int = 2,
    lr: float = 1e-4,
    epochs: int = 50,
    batch_size: int = 64,
    device: str = "cpu",
) -> PortfolioLSTM:
    """Train the LSTM model on training data.

    Args:
        X_train: Input features (n_samples, lookback, n_assets)
        y_train: Target returns (n_samples, n_assets)

    Returns:
        Trained PortfolioLSTM model.
    """
    model = PortfolioLSTM(n_assets, hidden_size=hidden_size, num_layers=num_layers).to(device)
    criterion = SharpeLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_t = torch.FloatTensor(X_train).to(device)
    y_t = torch.FloatTensor(y_train).to(device)

    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            weights = model(X_batch)
            loss = criterion(weights, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    model.eval()
    return model


def predict_weights(model: PortfolioLSTM, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Generate portfolio weights from input sequences.

    Returns:
        Array of shape (n_samples, n_assets) with weights summing to 1.
    """
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(device)
        weights = model(X_t).cpu().numpy()
    return weights

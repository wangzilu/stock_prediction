"""PyTorch deep models for stock prediction with MPS support.

Bypasses Qlib's built-in deep models (which don't support MPS)
by implementing ALSTM and Transformer directly with PyTorch,
using Qlib only for data loading.

All models implement the same interface:
    model = DeepModel(...)
    model.fit(X_train, y_train, X_valid, y_valid)
    pred = model.predict(X_test)
"""
import logging
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def get_device():
    """Get best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ALSTMNet(nn.Module):
    """Attention-LSTM network for stock prediction."""

    def __init__(self, d_feat=158, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        # Attention
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, features) → treat as (batch, 1, features) for single-step
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.rnn(x)  # (batch, seq, hidden)
        attn_weights = torch.softmax(self.attn(out), dim=1)
        context = (out * attn_weights).sum(dim=1)
        return self.fc(context).squeeze(-1)


class TransformerNet(nn.Module):
    """Transformer encoder for stock prediction."""

    def __init__(self, d_feat=158, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(d_feat, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.input_proj(x)
        x = self.encoder(x)
        x = x[:, -1, :]  # last step
        return self.fc(x).squeeze(-1)


class DeepModel:
    """Wrapper that trains a PyTorch model on Qlib-prepared data with MPS."""

    def __init__(
        self,
        net_class,
        net_kwargs=None,
        n_epochs=50,
        lr=1e-3,
        batch_size=2048,
        early_stop=10,
        device=None,
    ):
        self.net_class = net_class
        self.net_kwargs = net_kwargs or {}
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.device = device or get_device()
        self.model = None
        self.best_state = None

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        """Train the model.

        Args:
            X_train: np.ndarray (n_samples, n_features)
            y_train: np.ndarray (n_samples,)
            X_valid: optional validation features
            y_valid: optional validation labels
        """
        # Clean NaN/Inf
        train_mask = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]

        self.model = self.net_class(**self.net_kwargs).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        train_ds = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train),
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        # Validation
        if X_valid is not None and y_valid is not None:
            valid_mask = np.isfinite(X_valid).all(axis=1) & np.isfinite(y_valid)
            X_valid = X_valid[valid_mask]
            y_valid = y_valid[valid_mask]

        best_val_loss = float("inf")
        patience = 0

        logger.info(f"Training {self.net_class.__name__} on {self.device} ({len(X_train)} samples, {self.n_epochs} epochs)")

        for epoch in range(self.n_epochs):
            self.model.train()
            total_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)

                if not torch.isfinite(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / max(len(train_loader), 1)

            # Validation
            if X_valid is not None and len(X_valid) > 0:
                val_loss = self._eval_loss(X_valid, y_valid, criterion)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1

                if (epoch + 1) % 10 == 0:
                    logger.info(f"  Epoch {epoch+1}: train={avg_loss:.6f} valid={val_loss:.6f} best={best_val_loss:.6f}")

                if patience >= self.early_stop:
                    logger.info(f"  Early stop at epoch {epoch+1}")
                    break
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(f"  Epoch {epoch+1}: train={avg_loss:.6f}")

        # Load best weights
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        if self.device.type == "mps":
            torch.mps.synchronize()

    def _eval_loss(self, X, y, criterion):
        self.model.eval()
        with torch.no_grad():
            x_t = torch.FloatTensor(X).to(self.device)
            y_t = torch.FloatTensor(y).to(self.device)
            pred = self.model(x_t)
            loss = criterion(pred, y_t)
        return loss.item()

    def predict(self, X):
        """Predict scores.

        Args:
            X: np.ndarray (n_samples, n_features)

        Returns:
            np.ndarray (n_samples,) of predictions
        """
        self.model.eval()
        mask = np.isfinite(X).all(axis=1)
        result = np.full(len(X), np.nan)

        if mask.sum() == 0:
            return result

        with torch.no_grad():
            x_t = torch.FloatTensor(X[mask]).to(self.device)
            # Process in batches
            preds = []
            for i in range(0, len(x_t), self.batch_size):
                batch = x_t[i:i + self.batch_size]
                p = self.model(batch).cpu().numpy()
                preds.append(p)
            result[mask] = np.concatenate(preds)

        if self.device.type == "mps":
            torch.mps.synchronize()

        return result

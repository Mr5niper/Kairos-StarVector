# stock_forecast/models/lstm.py
"""Conditional LSTM regressor."""
import torch
import torch.nn as nn


class ConditionalLSTMRegressor(nn.Module):
    """
    LSTM over [target history | conditioning features], predicting the
    next value.

    Note the dropout guard: nn.LSTM ignores dropout when num_layers == 1
    and warns about it. Passing it through unchanged, as the original did,
    produced a warning on every single construction.
    """

    def __init__(self, in_dim: int = 1, cond_dim: int = 0, hidden: int = 64,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.total_dim = int(in_dim) + int(cond_dim)
        eff_dropout = float(dropout) if int(num_layers) > 1 else 0.0
        self.lstm = nn.LSTM(self.total_dim, int(hidden),
                            num_layers=int(num_layers), batch_first=True,
                            dropout=eff_dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(int(hidden)),
            nn.Linear(int(hidden), 1),
        )

    def forward(self, x_seq, cond_seq=None):
        x = torch.cat([x_seq, cond_seq], dim=-1) if (self.cond_dim and cond_seq is not None) else x_seq
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])

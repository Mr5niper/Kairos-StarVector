# stock_forecast/models/lstm.py
import torch
import torch.nn as nn

class ConditionalLSTMRegressor(nn.Module):
    def __init__(self, in_dim=1, cond_dim=0, hidden=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.total_dim = in_dim + cond_dim
        self.cond_dim = cond_dim
        self.lstm = nn.LSTM(self.total_dim, hidden, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x_seq, cond_seq=None):
        if self.cond_dim > 0 and cond_seq is not None:
            x = torch.cat([x_seq, cond_seq], dim=-1)
        else:
            x = x_seq
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
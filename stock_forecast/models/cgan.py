# stock_forecast/models/cgan.py
import torch
import torch.nn as nn
from torch import autograd

class Generator(nn.Module):
    def __init__(self, in_dim=1, cond_dim=0, hidden=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.total_dim = in_dim + cond_dim
        self.lstm = nn.LSTM(self.total_dim, hidden, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x_seq, cond_seq=None):
        if cond_seq is not None:
            x = torch.cat([x_seq, cond_seq], dim=-1)
        else:
            x = x_seq
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class Discriminator(nn.Module):
    def __init__(self, in_dim=1, cond_dim=0, hidden=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_dim = in_dim + 1 + cond_dim
        self.lstm = nn.LSTM(self.input_dim, hidden, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x_seq, y_next, cond_seq=None):
        B, T, _ = x_seq.size()
        y_rep = y_next.unsqueeze(1).repeat(1, T, 1)
        xy = torch.cat([x_seq, y_rep], dim=-1)
        if cond_seq is not None:
            xy = torch.cat([xy, cond_seq], dim=-1)
        out, _ = self.lstm(xy)
        return self.fc(out[:, -1, :])

def gradient_penalty(D, x_seq, real_y, fake_y, cond_seq=None, lambda_gp=10.0, device="cpu"):
    B = real_y.size(0)
    alpha = torch.rand(B, 1, device=device)
    interp = alpha * real_y + (1 - alpha) * fake_y
    interp.requires_grad_(True)
    d_interp = D(x_seq, interp, cond_seq)
    grads = autograd.grad(outputs=d_interp, inputs=interp,
                          grad_outputs=torch.ones_like(d_interp),
                          create_graph=True, retain_graph=True, only_inputs=True)[0]
    grads = grads.view(B, -1)
    gp = lambda_gp * ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp
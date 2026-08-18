# stock_forecast/models/cgan.py
"""Conditional WGAN-GP generator, critic and gradient penalty."""
import torch
import torch.nn as nn
from torch import autograd


def _dropout_for(num_layers: int, dropout: float) -> float:
    return float(dropout) if int(num_layers) > 1 else 0.0


class Generator(nn.Module):
    """
    Sequence-to-point generator.

    Takes a noise vector as well as the history, unlike the original,
    which was deterministic. A generator with no stochastic input cannot
    represent a distribution at all: it collapses to a point estimate and
    the adversarial objective loses its meaning.
    """

    def __init__(self, in_dim: int = 1, cond_dim: int = 0, hidden: int = 64,
                 num_layers: int = 2, dropout: float = 0.1, noise_dim: int = 8):
        super().__init__()
        self.noise_dim = int(noise_dim)
        self.lstm = nn.LSTM(int(in_dim) + int(cond_dim), int(hidden),
                            num_layers=int(num_layers), batch_first=True,
                            dropout=_dropout_for(num_layers, dropout))
        self.head = nn.Sequential(
            nn.Linear(int(hidden) + self.noise_dim, int(hidden)),
            nn.LeakyReLU(0.2),
            nn.Linear(int(hidden), 1),
        )

    def forward(self, x_seq, cond_seq=None, noise=None):
        x = torch.cat([x_seq, cond_seq], dim=-1) if cond_seq is not None else x_seq
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        if self.noise_dim > 0:
            if noise is None:
                noise = torch.randn(h.size(0), self.noise_dim, device=h.device)
            h = torch.cat([h, noise], dim=-1)
        return self.head(h)


class Discriminator(nn.Module):
    """
    Critic scoring (history, conditioning, candidate next value).

    No sigmoid on the output, and no batch normalisation anywhere: WGAN-GP
    requires an unbounded critic, and the gradient penalty is defined per
    sample, so any cross-sample normalisation invalidates it.
    """

    def __init__(self, in_dim: int = 1, cond_dim: int = 0, hidden: int = 64,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(int(in_dim) + 1 + int(cond_dim), int(hidden),
                            num_layers=int(num_layers), batch_first=True,
                            dropout=_dropout_for(num_layers, dropout))
        self.head = nn.Sequential(
            nn.Linear(int(hidden), int(hidden)),
            nn.LeakyReLU(0.2),
            nn.Linear(int(hidden), 1),
        )

    def forward(self, x_seq, y_next, cond_seq=None):
        b, t, _ = x_seq.size()
        y_rep = y_next.unsqueeze(1).repeat(1, t, 1)
        xy = torch.cat([x_seq, y_rep], dim=-1)
        if cond_seq is not None:
            xy = torch.cat([xy, cond_seq], dim=-1)
        out, _ = self.lstm(xy)
        return self.head(out[:, -1, :])


def gradient_penalty(critic, x_seq, real_y, fake_y, cond_seq=None,
                     lambda_gp: float = 10.0, device: str = "cpu"):
    """Two-sided penalty on the critic's gradient norm at interpolates."""
    b = real_y.size(0)
    alpha = torch.rand(b, 1, device=device)
    interp = (alpha * real_y + (1 - alpha) * fake_y).requires_grad_(True)
    score = critic(x_seq, interp, cond_seq)
    grads = autograd.grad(
        outputs=score, inputs=interp,
        grad_outputs=torch.ones_like(score),
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0].view(b, -1)
    return float(lambda_gp) * ((grads.norm(2, dim=1) - 1.0) ** 2).mean()

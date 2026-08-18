# stock_forecast/train_gan.py
"""Training loop for the conditional WGAN-GP."""
from typing import Dict, Tuple

import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .metrics import directional_accuracy, rmse
from .models.cgan import Discriminator, Generator, gradient_penalty


def _loader(x, c, y, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(np.asarray(x, dtype=np.float32)),
                       torch.from_numpy(np.asarray(c, dtype=np.float32)),
                       torch.from_numpy(np.asarray(y, dtype=np.float32)))
    return DataLoader(ds, batch_size=int(batch_size), shuffle=bool(shuffle),
                      drop_last=False)


def predict_mean(generator, x, c, device: str = "cpu", n_samples: int = 32) -> np.ndarray:
    """
    Average several noise draws to get a point forecast.

    The generator is stochastic by design, so a single forward pass is one
    sample from its predictive distribution rather than its expectation.
    Evaluating a WGAN with one draw per bar measures noise as much as
    skill; averaging over draws gives the conditional mean, which is what
    RMSE and direction accuracy are meant to be scored against.
    """
    generator.eval()
    xb = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
    cb = torch.from_numpy(np.asarray(c, dtype=np.float32)).to(device)
    acc = None
    with torch.no_grad():
        for _ in range(max(int(n_samples), 1)):
            out = generator(xb, cb).cpu().numpy().ravel()
            acc = out if acc is None else acc + out
    return acc / max(int(n_samples), 1)


def train_cwgan_gp(x_tr, c_tr, y_tr, x_va, c_va, y_va, cfg: Dict, y_scaler,
                   device: str = "cpu", log=None) -> Tuple[torch.nn.Module, torch.nn.Module, Dict]:
    """
    Train generator and critic, keeping the generator weights that scored
    best on validation.

    The critic is updated `crit_steps` times per generator step, as WGAN-GP
    requires: the gradient penalty only approximates the Lipschitz
    constraint if the critic stays near optimal.
    """
    def say(m):
        if log:
            log(m)

    x_tr = np.asarray(x_tr, dtype=np.float32)
    c_tr = np.asarray(c_tr, dtype=np.float32)
    if len(x_tr) < 8:
        raise RuntimeError(f"Only {len(x_tr)} training sequences; need at least 8.")

    in_dim, cond_dim = x_tr.shape[-1], c_tr.shape[-1]
    gen = Generator(in_dim=in_dim, cond_dim=cond_dim, hidden=int(cfg["hidden"]),
                    num_layers=int(cfg["layers"]),
                    dropout=float(cfg.get("dropout", 0.1))).to(device)
    crit = Discriminator(in_dim=in_dim, cond_dim=cond_dim, hidden=int(cfg["hidden"]),
                         num_layers=int(cfg["layers"]),
                         dropout=float(cfg.get("dropout", 0.1))).to(device)

    opt_g = torch.optim.Adam(gen.parameters(), lr=float(cfg["lr_g"]),
                             betas=tuple(cfg.get("betas_g", (0.5, 0.9))))
    opt_c = torch.optim.Adam(crit.parameters(), lr=float(cfg["lr_d"]),
                             betas=tuple(cfg.get("betas_d", (0.5, 0.9))))

    tr_dl = _loader(x_tr, c_tr, y_tr, cfg["batch_size"], True)
    metric = str(cfg.get("val_metric", "DPA")).upper()
    best = -np.inf if metric == "DPA" else np.inf
    best_g = copy.deepcopy(gen.state_dict())
    patience, bad = int(cfg.get("early_stopping", 0)), 0
    crit_steps = max(int(cfg.get("crit_steps", 3)), 1)
    lambda_gp = float(cfg.get("lambda_gp", 10.0))

    for epoch in range(int(cfg["epochs"])):
        gen.train()
        crit.train()
        for xb, cb, yb in tr_dl:
            xb, cb, yb = xb.to(device), cb.to(device), yb.to(device)

            for _ in range(crit_steps):
                opt_c.zero_grad()
                with torch.no_grad():
                    fake = gen(xb, cb)
                d_real = crit(xb, yb, cb).mean()
                d_fake = crit(xb, fake, cb).mean()
                gp = gradient_penalty(crit, xb, yb, fake, cb, lambda_gp, device)
                (d_fake - d_real + gp).backward()
                opt_c.step()

            opt_g.zero_grad()
            (-crit(xb, gen(xb, cb), cb).mean()).backward()
            opt_g.step()

        p = y_scaler.inverse_transform(predict_mean(gen, x_va, c_va, device, 16)).ravel()
        t = y_scaler.inverse_transform(np.asarray(y_va, dtype=np.float32)).ravel()
        n = min(len(p), len(t))
        v_dpa, v_rmse = directional_accuracy(t[:n], p[:n]), rmse(t[:n], p[:n])
        score = v_dpa if metric == "DPA" else v_rmse
        better = (score > best) if metric == "DPA" else (score < best)

        if better:
            best = score
            best_g = copy.deepcopy(gen.state_dict())
            bad = 0
        else:
            bad += 1

        say(f"[cWGAN-GP] epoch {epoch+1}/{cfg['epochs']} rmse={v_rmse:.5f} "
            f"dpa={v_dpa:.4f} best={best:.4f}")

        if patience and bad >= patience:
            say("[cWGAN-GP] early stop")
            break

    gen.load_state_dict(best_g)
    return gen, crit, {"best_val": float(best), "metric": metric}

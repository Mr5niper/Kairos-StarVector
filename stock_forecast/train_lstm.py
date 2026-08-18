# stock_forecast/train_lstm.py
"""Training loop for the conditional LSTM."""
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .metrics import directional_accuracy, rmse
from .models.lstm import ConditionalLSTMRegressor


def _loader(x, c, y, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(np.asarray(x, dtype=np.float32)),
                       torch.from_numpy(np.asarray(c, dtype=np.float32)),
                       torch.from_numpy(np.asarray(y, dtype=np.float32)))
    return DataLoader(ds, batch_size=int(batch_size), shuffle=bool(shuffle),
                      drop_last=False)


def train_lstm(x_tr, c_tr, y_tr, x_va, c_va, y_va, cfg: Dict, y_scaler,
               device: str = "cpu", log=None) -> Tuple[nn.Module, Dict]:
    """
    Train with early stopping on a validation metric, restoring the best
    weights before returning.

    Two behaviours worth knowing about. First, when the metric is DPA
    (direction accuracy) the loop keeps the epoch with the best direction
    score, not the best loss; those rarely coincide, and direction is what
    a trading signal needs. Second, ties do not overwrite the incumbent,
    so the earliest of several equally good epochs wins. That matters
    because DPA on a small validation block takes only a handful of
    distinct values and ties are common; without the guard the loop drifts
    to the last tied epoch, which is usually the more overfitted one.
    """
    def say(m):
        if log:
            log(m)

    x_tr = np.asarray(x_tr, dtype=np.float32)
    c_tr = np.asarray(c_tr, dtype=np.float32)
    if len(x_tr) < 4:
        raise RuntimeError(f"Only {len(x_tr)} training sequences; need at least 4.")

    metric = str(cfg.get("val_metric", "DPA")).upper()
    model = ConditionalLSTMRegressor(
        in_dim=x_tr.shape[-1], cond_dim=c_tr.shape[-1],
        hidden=int(cfg["hidden"]), num_layers=int(cfg["layers"]),
        dropout=float(cfg.get("dropout", 0.1)),
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    loss_fn = nn.MSELoss()
    tr_dl = _loader(x_tr, c_tr, y_tr, cfg["batch_size"], True)
    va_dl = _loader(x_va, c_va, y_va, cfg["batch_size"], False)

    best = -np.inf if metric == "DPA" else np.inf
    best_state, bad = None, 0
    patience = int(cfg.get("early_stopping", 0))

    for epoch in range(int(cfg["epochs"])):
        model.train()
        for xb, cb, yb in tr_dl:
            xb, cb, yb = xb.to(device), cb.to(device), yb.to(device)
            loss = loss_fn(model(xb, cb), yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for xb, cb, yb in va_dl:
                preds.append(model(xb.to(device), cb.to(device)).cpu().numpy())
                targets.append(yb.numpy())
        if not preds:
            break
        p = y_scaler.inverse_transform(np.concatenate(preds)).ravel()
        t = y_scaler.inverse_transform(np.concatenate(targets)).ravel()

        v_dpa, v_rmse = directional_accuracy(t, p), rmse(t, p)
        score = v_dpa if metric == "DPA" else v_rmse
        better = (score > best) if metric == "DPA" else (score < best)

        if better:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        say(f"[LSTM] epoch {epoch+1}/{cfg['epochs']} rmse={v_rmse:.5f} "
            f"dpa={v_dpa:.4f} best={best:.4f}")

        if patience and bad >= patience:
            say("[LSTM] early stop")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val": float(best), "metric": metric}

# stock_forecast/train_lstm.py
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from stock_forecast.models.lstm import ConditionalLSTMRegressor
from stock_forecast.metrics import directional_accuracy, rmse

def get_lstm_loader(X, C, y, batch_size, shuffle):
    return DataLoader(TensorDataset(torch.tensor(X), torch.tensor(C), torch.tensor(y)),
                      batch_size=batch_size, shuffle=shuffle, drop_last=False)

def train_lstm(X_tr, C_tr, y_tr, X_va, C_va, y_va, cfg: Dict, y_scaler, device: str = "cpu") -> Tuple[torch.nn.Module, Dict]:
    in_dim = X_tr.shape[-1]
    cond_dim = C_tr.shape[-1]
    val_metric = cfg.get('val_metric', 'RMSE')

    model = ConditionalLSTMRegressor(in_dim=in_dim, cond_dim=cond_dim,
                                     hidden=cfg['hidden'], num_layers=cfg['layers'],
                                     dropout=cfg['dropout']).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    loss_fn = nn.MSELoss()

    tr_dl = get_lstm_loader(X_tr, C_tr, y_tr, cfg['batch_size'], shuffle=True)
    va_dl = get_lstm_loader(X_va, C_va, y_va, cfg['batch_size'], shuffle=False)

    best = -np.inf if val_metric == 'DPA' else np.inf
    best_state = None
    patience = cfg.get('early_stopping', 0)
    bad = 0

    for epoch in range(cfg['epochs']):
        model.train()
        for xb, cb, yb in tr_dl:
            xb = xb.to(device); cb = cb.to(device); yb = yb.to(device)
            pred = model(xb, cb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # --- Validation ---
        model.eval()
        preds_s, targets_s = [], []
        with torch.no_grad():
            for xb, cb, yb in va_dl:
                xb = xb.to(device); cb = cb.to(device); yb = yb.to(device)
                yhat = model(xb, cb)
                preds_s.append(yhat.cpu().numpy())
                targets_s.append(yb.cpu().numpy())

        preds_s = np.concatenate(preds_s)
        targets_s = np.concatenate(targets_s)
        preds_raw = y_scaler.inverse_transform(preds_s).ravel()
        targets_raw = y_scaler.inverse_transform(targets_s).ravel()

        val_dpa  = directional_accuracy(targets_raw, preds_raw)
        val_rmse = rmse(targets_raw, preds_raw)

        if val_metric == 'DPA':
            is_better = val_dpa > best
            score = val_dpa
        else:
            is_better = val_rmse < best
            score = val_rmse

        if is_better:
            best = score
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        print(f"[LSTM+C] Epoch {epoch+1}/{cfg['epochs']} VAL_RMSE={val_rmse:.4f} VAL_DPA={val_dpa:.4f} (Best {val_metric}={best:.4f})")
        if patience and bad >= patience:
            print("[LSTM+C] Early stopping")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val": float(best)}
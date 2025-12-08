# stock_forecast/train_gan.py
from typing import Dict, Tuple
import copy
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from stock_forecast.models.cgan import Generator, Discriminator, gradient_penalty
from stock_forecast.metrics import rmse, directional_accuracy

def get_cgan_loader(X, C, y, batch_size, shuffle):
    return DataLoader(TensorDataset(torch.tensor(X), torch.tensor(C), torch.tensor(y)),
                      batch_size=batch_size, shuffle=shuffle, drop_last=False)

def train_cwgan_gp(X_tr, C_tr, y_tr, X_va, C_va, y_va, cfg: Dict, y_scaler, device: str = "cpu"
                  ) -> Tuple[torch.nn.Module, torch.nn.Module, Dict]:
    in_dim = X_tr.shape[-1]
    cond_dim = C_tr.shape[-1]
    G = Generator(in_dim=in_dim, cond_dim=cond_dim,
                  hidden=cfg['hidden'], num_layers=cfg['layers'], dropout=cfg['dropout']).to(device)
    D = Discriminator(in_dim=in_dim, cond_dim=cond_dim,
                      hidden=cfg['hidden'], num_layers=cfg['layers'], dropout=cfg['dropout']).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=cfg['lr_g'], betas=tuple(cfg['betas_g']))
    opt_D = torch.optim.Adam(D.parameters(), lr=cfg['lr_d'], betas=tuple(cfg['betas_d']))

    tr_dl = get_cgan_loader(X_tr, C_tr, y_tr, cfg['batch_size'], shuffle=True)
    va_dl = get_cgan_loader(X_va, C_va, y_va, cfg['batch_size'], shuffle=False)

    val_metric = cfg.get('val_metric', 'DPA')
    best = -np.inf if val_metric == 'DPA' else np.inf
    best_G = copy.deepcopy(G.state_dict())
    patience = cfg.get('early_stopping', 0)
    bad = 0

    for epoch in range(cfg['epochs']):
        G.train(); D.train()
        for xb, cb, yb in tr_dl:
            xb = xb.to(device); cb = cb.to(device); yb = yb.to(device)

            # Train D
            for _ in range(cfg['crit_steps']):
                D.zero_grad()
                y_fake = G(xb, cb).detach()
                d_real = D(xb, yb, cb).mean()
                d_fake = D(xb, y_fake, cb).mean()
                gp = gradient_penalty(D, xb, yb, y_fake, cb, lambda_gp=cfg['lambda_gp'], device=device)
                loss_D = d_fake - d_real + gp
                loss_D.backward()
                opt_D.step()

            # Train G
            G.zero_grad()
            y_fake = G(xb, cb)
            d_fake = D(xb, y_fake, cb).mean()
            loss_G = -d_fake
            loss_G.backward()
            opt_G.step()

        # Validation + Early stopping
        G.eval()
        with torch.no_grad():
            preds_s, targets_s = [], []
            for xb, cb, yb in va_dl:
                xb = xb.to(device); cb = cb.to(device)
                yhat = G(xb, cb)
                preds_s.append(yhat.cpu().numpy())
                targets_s.append(yb.numpy())
            preds_s = np.concatenate(preds_s)
            targets_s = np.concatenate(targets_s)
            preds_raw = y_scaler.inverse_transform(preds_s).ravel()
            targets_raw = y_scaler.inverse_transform(targets_s).ravel()
            v_rmse = rmse(targets_raw, preds_raw)
            v_dpa  = directional_accuracy(targets_raw, preds_raw)

            if val_metric == 'DPA':
                score = v_dpa; better = score > best
            else:
                score = v_rmse; better = score < best

            if better:
                best = score
                best_G = copy.deepcopy(G.state_dict())
                bad = 0
            else:
                bad += 1

            print(f"[cWGAN-GP] Epoch {epoch+1}/{cfg['epochs']} VAL_RMSE={v_rmse:.4f} VAL_DPA={v_dpa:.4f} (Best {val_metric}={best:.4f})")
            if patience and bad >= patience:
                print("[cWGAN-GP] Early stopping")
                break

    G.load_state_dict(best_G)
    return G, D, {"best_val": float(best)}
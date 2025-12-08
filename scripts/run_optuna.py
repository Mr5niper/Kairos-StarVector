# scripts/run_optuna.py
import os
import yaml
import optuna
import numpy as np
import torch

from stock_forecast.utils import set_seed
from stock_forecast.dataset import fetch_ohlc_yf, make_target, assemble_conditional, scale_fit_transform, build_sequences
from stock_forecast.splits import rolling_windows
from stock_forecast.train_lstm import train_lstm
from stock_forecast.train_gan import train_cwgan_gp
from stock_forecast.metrics import directional_accuracy

def score_window(cfg, y_series, cond_df, tr, va, te, trial):
    y_tr_raw = y_series.values[tr[0]:tr[1]]
    y_va_raw = y_series.values[va[0]:va[1]]
    y_te_raw = y_series.values[te[0]:te[1]]
    C_tr_raw = cond_df.values[tr[0]:tr[1], :]
    C_va_raw = cond_df.values[va[0]:va[1], :]
    C_te_raw = cond_df.values[te[0]:te[1], :]

    y_scaler, y_tr, y_va, y_te = scale_fit_transform(y_tr_raw, y_va_raw, y_te_raw, is_2d=False)
    c_scaler, C_tr, C_va, C_te = scale_fit_transform(C_tr_raw, C_va_raw, C_te_raw, is_2d=True)
    (X_tr, Cseq_tr, t_tr) = build_sequences(y_tr, C_tr, cfg['data']['seq_len'])
    (X_va, Cseq_va, t_va) = build_sequences(np.concatenate([y_tr[-cfg['data']['seq_len']:], y_va]),
                                           np.concatenate([C_tr[-cfg['data']['seq_len']:], C_va], axis=0),
                                           cfg['data']['seq_len'])

    # hyperparams
    lstm_hidden = trial.suggest_categorical("lstm_hidden", [32, 64, 96])
    lstm_layers = trial.suggest_int("lstm_layers", 1, 3)
    lstm_lr     = trial.suggest_float("lstm_lr", 5e-4, 2e-3, log=True)
    gan_hidden  = trial.suggest_categorical("gan_hidden", [32, 64, 96])
    gan_layers  = trial.suggest_int("gan_layers", 1, 3)
    gan_lr_g    = trial.suggest_float("gan_lr_g", 1e-4, 5e-4, log=True)
    gan_lr_d    = trial.suggest_float("gan_lr_d", 1e-4, 5e-4, log=True)

    lstm_cfg = cfg['model']['lstm'].copy()
    lstm_cfg['hidden'] = lstm_hidden
    lstm_cfg['layers'] = lstm_layers
    lstm_cfg['lr']     = lstm_lr

    gan_cfg = cfg['model']['gan'].copy()
    gan_cfg['hidden'] = gan_hidden
    gan_cfg['layers'] = gan_layers
    gan_cfg['lr_g']   = gan_lr_g
    gan_cfg['lr_d']   = gan_lr_d

    # Train LSTM
    model, _ = train_lstm(X_tr, Cseq_tr, t_tr, X_va, Cseq_va, t_va, lstm_cfg, y_scaler, device="cpu")
    with torch.no_grad():
        yhat_s = model(torch.tensor(X_va), torch.tensor(Cseq_va)).numpy().ravel()
    yhat = y_scaler.inverse_transform(yhat_s.reshape(-1,1)).ravel()
    y_true = y_va_raw[-len(yhat):]
    dpa_lstm = directional_accuracy(y_true, yhat)

    # Train GAN
    G, D, _ = train_cwgan_gp(X_tr, Cseq_tr, t_tr, X_va, Cseq_va, t_va, gan_cfg, y_scaler, device="cpu")
    with torch.no_grad():
        yhat_g_s = G(torch.tensor(X_va), torch.tensor(Cseq_va)).numpy().ravel()
    yhat_g = y_scaler.inverse_transform(yhat_g_s.reshape(-1,1)).ravel()
    dpa_gan = directional_accuracy(y_true, yhat_g)

    return 0.4*dpa_lstm + 0.6*dpa_gan

def objective(trial, cfg):
    df = fetch_ohlc_yf(cfg['data']['ticker'], start=cfg['data']['start'], end=cfg['data']['end'])
    y_series = make_target(df['Close'], mode=cfg['data']['target'])
    cond_df, cond_dim = assemble_conditional(cfg['features']['mode'], df, cfg['features'])
    idx = y_series.index.intersection(cond_df.index)
    y_series = y_series.loc[idx]
    cond_df = cond_df.loc[idx]
    N = len(y_series)

    windows = list(rolling_windows(N, cfg['data']['seq_len'], cfg['splits']['train_len'], cfg['splits']['val_len'], cfg['splits']['test_len'], cfg['splits']['step']))
    K = min(cfg['tuning']['windows'], len(windows))
    if K == 0:
        return 0.0

    scores = []
    for i in range(K):
        tr, va, te = windows[i]
        try:
            s = score_window(cfg, y_series, cond_df, tr, va, te, trial)
            scores.append(s)
        except Exception:
            continue

    if len(scores) == 0:
        return 0.0
    return float(np.mean(scores))

def main(cfg_path="configs/default.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg.get("seed", 42))

    if not cfg.get("tuning", {}).get("enable", False):
        print("Tuning disabled in config.")
        return

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, cfg), n_trials=cfg['tuning']['n_trials'])

    print("Best Trial:", study.best_trial.value)
    print("Best Params:", study.best_trial.params)

if __name__ == "__main__":
    main()
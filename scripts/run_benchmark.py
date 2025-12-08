# scripts/run_benchmark.py
import os
import json
import yaml
import numpy as np
import pandas as pd
import torch

from stock_forecast.utils import set_seed, ensure_dir, device_info
from stock_forecast.dataset import (
    fetch_ohlc_yf, make_target, assemble_conditional,
    scale_fit_transform, build_sequences
)
from stock_forecast.splits import rolling_windows
from stock_forecast.models.arima import arima_forecast
from stock_forecast.train_lstm import train_lstm
from stock_forecast.train_gan import train_cwgan_gp
from stock_forecast.eval import evaluate_predictions
from stock_forecast.metrics import diebold_mariano
from stock_forecast.backtest import simple_long_short
from stock_forecast.meta_labeling import build_meta_features, build_meta_labels, train_meta_clf, apply_meta_clf

def predict_deep(model, X, C, device="cpu"):
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X).to(device)
        cb = torch.tensor(C).to(device) if C is not None else None
        yhat = model(xb, cb).cpu().numpy().ravel()
    return yhat

def get_cond_2d(X_seq, C_seq):
    return np.concatenate([X_seq[:, -1, :], C_seq[:, -1, :]], axis=1)

def run_window(df_aligned, y_series, cond_df, cfg, device, window_id, tr, va, te):
    results = {}
    # slices
    y_tr_raw = y_series.values[tr[0]:tr[1]]
    y_va_raw = y_series.values[va[0]:va[1]]
    y_te_raw = y_series.values[te[0]:te[1]]
    C_tr_raw = cond_df.values[tr[0]:tr[1], :]
    C_va_raw = cond_df.values[va[0]:va[1], :]
    C_te_raw = cond_df.values[te[0]:te[1], :]

    # scale target and cond
    y_scaler, y_tr, y_va, y_te = scale_fit_transform(y_tr_raw, y_va_raw, y_te_raw, is_2d=False)
    c_scaler, C_tr, C_va, C_te = scale_fit_transform(C_tr_raw, C_va_raw, C_te_raw, is_2d=True)

    # sequences
    seq_len = cfg['data']['seq_len']
    (X_tr, Cseq_tr, t_tr) = build_sequences(y_tr, C_tr, seq_len)
    (X_va, Cseq_va, t_va) = build_sequences(np.concatenate([y_tr[-seq_len:], y_va]),
                                           np.concatenate([C_tr[-seq_len:], C_va], axis=0),
                                           seq_len)
    (X_te, Cseq_te, t_te) = build_sequences(np.concatenate([y_tr[-seq_len:], y_va, y_te]),
                                           np.concatenate([C_tr[-seq_len:], C_va, C_te], axis=0),
                                           seq_len)

    # align test outputs to length
    test_len = len(y_te_raw)
    X_te = X_te[-test_len:]; Cseq_te = Cseq_te[-test_len:]; t_te = t_te[-test_len:]

    # --- ARIMA (raw) ---
    try:
        ar_preds = []
        arr = y_tr_raw.copy()
        for i in range(len(y_te_raw)):
            yhat = arima_forecast(arr, order=(3,1,2))
            ar_preds.append(yhat)
            arr = np.concatenate([arr, [y_te_raw[i]]])
        ar_preds = np.array(ar_preds)
    except Exception as e:
        print(f"[ARIMA] Error: {e}")
        ar_preds = np.zeros_like(y_te_raw)

    # --- LSTM (DPA-optimized) ---
    try:
        lstm_model, _ = train_lstm(X_tr, Cseq_tr, t_tr, X_va, Cseq_va, t_va, cfg['model']['lstm'], y_scaler, device=device)
        yhat_lstm_s = predict_deep(lstm_model, X_te, Cseq_te, device)
        yhat_lstm = y_scaler.inverse_transform(yhat_lstm_s.reshape(-1,1)).ravel()
    except Exception as e:
        print(f"[LSTM] Error: {e}")
        yhat_lstm = np.zeros_like(y_te_raw)

    # --- cWGAN-GP ---
    try:
        G, D, _ = train_cwgan_gp(X_tr, Cseq_tr, t_tr, X_va, Cseq_va, t_va, cfg['model']['gan'], y_scaler, device=device)
        yhat_gan_s = predict_deep(G, X_te, Cseq_te, device)
        yhat_gan = y_scaler.inverse_transform(yhat_gan_s.reshape(-1,1)).ravel()
    except Exception as e:
        print(f"[cWGAN-GP] Error: {e}")
        yhat_gan = np.zeros_like(y_te_raw)

    # --- Residual Fusion (ablation compare) ---
    yhat_res = None
    try:
        if cfg['model']['residual']['enabled']:
            import lightgbm as lgb
            params = cfg['model']['residual']['lgbm_params']
            F_tr = get_cond_2d(X_tr, Cseq_tr)
            F_va = get_cond_2d(X_va, Cseq_va)
            F_te = get_cond_2d(X_te, Cseq_te)
            lgbm = lgb.LGBMRegressor(**params, random_state=42)
            lgbm.fit(F_tr, t_tr.ravel(), eval_set=[(F_va, t_va.ravel())], verbose=False)
            yhat_lgbm_te = lgbm.predict(F_te).reshape(-1,1)
            yhat_lgbm_tr = lgbm.predict(F_tr).reshape(-1,1)
            res_tr = t_tr - yhat_lgbm_tr
            G_res, _, _ = train_cwgan_gp(X_tr, Cseq_tr, res_tr, X_va, Cseq_va, t_va, cfg['model']['gan'], y_scaler, device=device)
            res_hat_te_s = predict_deep(G_res, X_te, Cseq_te, device).reshape(-1,1)
            yhat_res_s = yhat_lgbm_te + res_hat_te_s
            yhat_res = y_scaler.inverse_transform(yhat_res_s).ravel()
    except Exception as e:
        print(f"[Residual] Error: {e}")
        yhat_res = None

    # --- Meta-Labeling (filter GAN trades) ---
    yhat_gan_filtered = yhat_gan.copy()
    try:
        if cfg['meta_labeling']['enabled']:
            # Train meta on VALIDATION set using GAN preds
            preds_val_s = predict_deep(G, X_va, Cseq_va, device)
            preds_val = y_scaler.inverse_transform(preds_val_s.reshape(-1,1)).ravel()
            y_true_val = y_series.values[va[0]:va[1]][-len(preds_val):]
            C_last_val = Cseq_va[:, -1, :]     # (N_val, F)
            meta_X_val = build_meta_features(C_last_val, preds_val, returns_vol=None)
            meta_y_val = build_meta_labels(y_true_val, preds_val, abs_threshold=cfg['meta_labeling']['threshold_abs_pred'])
            # Train meta classifier
            clf = train_meta_clf(meta_X_val, meta_y_val, cfg['meta_labeling']['lgbm_params'])
            # Apply on test set
            C_last_te = Cseq_te[:, -1, :]
            meta_X_te = build_meta_features(C_last_te, yhat_gan, returns_vol=None)
            yhat_gan_filtered, accept_prob = apply_meta_clf(clf, meta_X_te, yhat_gan)
    except Exception as e:
        print(f"[Meta] Error: {e}")
        yhat_gan_filtered = yhat_gan

    # Evaluate
    y_true = y_te_raw.ravel()
    eval_ar    = evaluate_predictions(y_true, ar_preds)
    eval_lstm  = evaluate_predictions(y_true, yhat_lstm)
    eval_gan   = evaluate_predictions(y_true, yhat_gan)
    eval_gan_f = evaluate_predictions(y_true, yhat_gan_filtered)
    eval_res   = evaluate_predictions(y_true, yhat_res) if yhat_res is not None else {}

    # Backtests
    tc = cfg['backtest']['tc_bps']
    bt_ar    = simple_long_short(y_true, ar_preds, tc_bps=tc)
    bt_lstm  = simple_long_short(y_true, yhat_lstm, tc_bps=tc)
    bt_gan   = simple_long_short(y_true, yhat_gan, tc_bps=tc)
    bt_gan_f = simple_long_short(y_true, yhat_gan_filtered, tc_bps=tc)
    bt_res   = simple_long_short(y_true, yhat_res, tc_bps=tc) if yhat_res is not None else {}

    # DM tests
    dm_gan_vs_lstm = diebold_mariano(y_true - yhat_lstm, y_true - yhat_gan)
    dm_gan_vs_arima= diebold_mariano(y_true - ar_preds,   y_true - yhat_gan)

    ablation = {}
    if cfg['ablation'].get('residual_compare', False) and yhat_res is not None:
        ablation['residual_beats_gan_DPA'] = float(eval_res['DPA'] - eval_gan['DPA'])
    if cfg['ablation'].get('meta_compare', False):
        ablation['meta_lifts_gan_DPA'] = float(eval_gan_f['DPA'] - eval_gan['DPA'])
        ablation['meta_lifts_gan_Sharpe'] = float(bt_gan_f['sharpe'] - bt_gan['sharpe'])

    results = {
        "ARIMA": {**eval_ar, **{f"BT_{k}": v for k, v in bt_ar.items()}},
        "LSTM_COND": {**eval_lstm, **{f"BT_{k}": v for k, v in bt_lstm.items()}},
        "CWGAN_GP_COND": {**eval_gan, **{f"BT_{k}": v for k, v in bt_gan.items()}},
        "CWGAN_GP_COND_META": {**eval_gan_f, **{f"BT_{k}": v for k, v in bt_gan_f.items()}},
        "DM_GAN_vs_LSTM": {"stat": dm_gan_vs_lstm[0], "p": dm_gan_vs_lstm[1]},
        "DM_GAN_vs_ARIMA": {"stat": dm_gan_vs_arima[0], "p": dm_gan_vs_arima[1]},
        "ABLATION": ablation
    }
    if yhat_res is not None:
        results["RESIDUAL_FUSION"] = {**eval_res, **{f"BT_{k}": v for k, v in bt_res.items()}}

    return results, {
        "y_true": y_true,
        "y_arima": ar_preds,
        "y_lstm": yhat_lstm,
        "y_gan": yhat_gan,
        "y_gan_filtered": yhat_gan_filtered,
        "y_res": yhat_res,
        "dates": y_series.index[te[0]:te[1]]
    }

def aggregate_with_ci(results_all, model_key):
    keys = set()
    for r in results_all:
        if model_key in r:
            keys.update(r[model_key].keys())
    keys = list(keys)
    stats = {}
    for k in keys:
        vals = [r[model_key][k] for r in results_all if model_key in r and k in r[model_key]]
        if len(vals) == 0: 
            continue
        stats[k] = {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals)),
            "ci95": (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
        }
    return stats

def main(cfg_path="configs/default.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] Using {device.upper()} - {device_info()}")
    out_dir = cfg['eval']['out_dir']
    ensure_dir(out_dir)

    # Data and features
    df = fetch_ohlc_yf(cfg['data']['ticker'], start=cfg['data']['start'], end=cfg['data']['end'])
    y_series = make_target(df['Close'], mode=cfg['data']['target'])
    cond_df, cond_dim = assemble_conditional(cfg['features']['mode'], df, cfg['features'])
    idx = y_series.index.intersection(cond_df.index)
    y_series = y_series.loc[idx]
    cond_df = cond_df.loc[idx]
    df_aligned = df.loc[idx]
    print(f"[Data] N={len(y_series)} cond_dim={cond_dim}")

    N = len(y_series.values)
    results_all = []
    preds_saved = []

    for i, (tr, va, te) in enumerate(rolling_windows(N, cfg['data']['seq_len'], cfg['splits']['train_len'], cfg['splits']['val_len'], cfg['splits']['test_len'], cfg['splits']['step']), start=1):
        print(f"\n=== Window {i}: {df_aligned.index[te[0]]:%Y-%m-%d} to {df_aligned.index[te[1]-1]:%Y-%m-%d} ===")
        try:
            window_results, preds = run_window(df_aligned, y_series, cond_df, cfg, device, i, tr, va, te)
            window_results["window"] = {"start": str(df_aligned.index[te[0]]), "end": str(df_aligned.index[te[1]-1])}
            results_all.append(window_results)
            preds_saved.append(preds)

            if cfg['eval'].get("save_predictions", True):
                out_csv = os.path.join(out_dir, f"preds_window_{i}.csv")
                df_out = {
                    "Date": preds["dates"],
                    "y_true": preds["y_true"],
                    "y_arima": preds["y_arima"],
                    "y_lstm": preds["y_lstm"],
                    "y_gan": preds["y_gan"],
                    "y_gan_filtered": preds["y_gan_filtered"]
                }
                if preds["y_res"] is not None:
                    df_out["y_residual"] = preds["y_res"]
                pd.DataFrame(df_out).to_csv(out_csv, index=False)
        except Exception as e:
            print(f"[Window {i}] Error, skipping: {e}")
            continue

    # Aggregate with CI
    print("\n=== AVERAGED OVER WINDOWS (with 95% CI) ===")
    summary = {
        "ARIMA": aggregate_with_ci(results_all, "ARIMA"),
        "LSTM_COND": aggregate_with_ci(results_all, "LSTM_COND"),
        "CWGAN_GP_COND": aggregate_with_ci(results_all, "CWGAN_GP_COND"),
        "CWGAN_GP_COND_META": aggregate_with_ci(results_all, "CWGAN_GP_COND_META"),
        "RESIDUAL_FUSION": aggregate_with_ci(results_all, "RESIDUAL_FUSION"),
        "DM_GAN_vs_LSTM": aggregate_with_ci(results_all, "DM_GAN_vs_LSTM"),
        "DM_GAN_vs_ARIMA": aggregate_with_ci(results_all, "DM_GAN_vs_ARIMA"),
        "ABLATION": aggregate_with_ci(results_all, "ABLATION")
    }
    for k, v in summary.items():
        print(k, v)

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
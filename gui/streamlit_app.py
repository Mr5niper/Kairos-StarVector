# gui/streamlit_app.py
import os, sys, traceback, io, json, time
from datetime import datetime, date
from typing import Dict, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from stock_forecast.utils import set_seed, ensure_dir, device_info
from stock_forecast.dataset import (
    fetch_ohlc_yf, make_target, assemble_conditional,
    scale_fit_transform, build_sequences, get_raw_planetary_positions
)
from stock_forecast.splits import rolling_windows
from stock_forecast.models.arima import arima_forecast
from stock_forecast.train_lstm import train_lstm
from stock_forecast.train_gan import train_cwgan_gp
from stock_forecast.eval import evaluate_predictions
from stock_forecast.metrics import diebold_mariano
from stock_forecast.backtest import simple_long_short
from stock_forecast.meta_labeling import (
    build_meta_features, build_meta_labels, safe_train_meta_clf, apply_meta_clf
)
from stock_forecast.gann_grid import build_overlay_shapes, PLANETS
st.set_page_config(page_title="Market Forecast Lab", layout="wide")
st.markdown("""
<style>
[data-testid="stSidebar"] { min-width: 340px; max-width: 340px; }
.kpi-card { padding: 10px 15px; border-radius: 8px; border: 1px solid #2b2b2b; background: rgba(255,255,255,0.03);}
.kpi-title { font-size: 0.85rem; color: #aaa; margin-bottom: 2px;}
.kpi-value { font-size: 1.3rem; font-weight: 700;}
.run-header { padding: 8px 12px; border-left: 4px solid #2E86DE; background: rgba(46,134,222,.08); margin-bottom: 8px;}
</style>
""", unsafe_allow_html=True)
@st.cache_data(show_spinner=False, max_entries=8)
def _cached_planets(date_key: tuple):
    idx = pd.to_datetime(list(date_key))
    return get_raw_planetary_positions(pd.DatetimeIndex(idx))
def _key_from_dates(dates: pd.DatetimeIndex) -> tuple:
    return tuple(pd.DatetimeIndex(dates).strftime("%Y-%m-%d"))
def predict_deep(model, X, C, device="cpu"):
    import torch
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X).to(device)
        cb = torch.tensor(C).to(device) if C is not None else None
        yhat = model(xb, cb).cpu().numpy().ravel()
    return yhat
def get_cond_2d(X_seq, C_seq):
    return np.concatenate([X_seq[:, -1, :], C_seq[:, -1, :]], axis=1)
def kpi_card(title: str, value: str):
    st.markdown(f"""<div class="kpi-card"><div class="kpi-title">{title}</div><div class="kpi-value">{value}</div></div>""", unsafe_allow_html=True)
def plot_predictions(df_plot: pd.DataFrame, model_cols: List[str]):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot["Date"], y=df_plot["True"], mode="lines", name="True", line=dict(color="#ddd")))
    palette = ["#0E76A8", "#E67E22", "#27AE60", "#8E44AD", "#C0392B", "#16A085"]
    for i, col in enumerate(model_cols):
        fig.add_trace(go.Scatter(x=df_plot["Date"], y=df_plot[col], mode="lines", name=col, line=dict(color=palette[i % len(palette)])))
    fig.update_layout(legend=dict(orientation="h"), height=450, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")
def plot_firm_holdings(df_plot: pd.DataFrame, model_cols: List[str]):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot["Date"], y=df_plot["True"], mode="lines", name="Market (True)", line=dict(color="#FFFFFF", width=3)))
    colors = ["rgba(14,118,168,0.40)","rgba(230,126,34,0.40)","rgba(39,174,96,0.40)","rgba(142,68,173,0.40)","rgba(192,57,43,0.40)"]
    lines  = ["#0E76A8","#E67E22","#27AE60","#8E44AD","#C0392B"]
    for i, col in enumerate(model_cols):
        fig.add_trace(go.Scatter(x=df_plot["Date"], y=df_plot[col], mode="lines", name=col, stackgroup="one",
                                 line=dict(width=1, color=lines[i % len(lines)]), fillcolor=colors[i % len(colors)]))
    fig.update_layout(legend=dict(orientation="h"), height=500, margin=dict(l=0,r=0,t=10,b=0), hovermode="x unified")
    st.plotly_chart(fig, width="stretch")
def planet_price_traces(dates: pd.DatetimeIndex, close: pd.Series, planets: List[str], harmonics: List[int], deg_to_price: float):
    astro = _cached_planets(_key_from_dates(pd.DatetimeIndex(dates)))
    traces = []
    avg = float(close.mean())
    for i, pl in enumerate(planets):
        deg = astro[pl].values.astype(float)
        base = deg * float(deg_to_price)
        offset = avg - float(base.mean())
        color_cycle = ["#e74c3c","#f39c12","#8e44ad","#2980b9","#16a085","#c0392b"]
        for h in harmonics:
            y = base + offset + float(h)*360.0*float(deg_to_price)
            traces.append(go.Scatter(x=dates, y=y, mode="lines", name=f"{pl} h{h}",
                                     line=dict(width=1, color=color_cycle[i % len(color_cycle)]),
                                     opacity=0.45, showlegend=True))
    return traces
def plot_equity_curve(y_true: np.ndarray, preds_map: Dict[str, np.ndarray], tc_bps: float):
    fig = go.Figure()
    for name, yhat in preds_map.items():
        bt = simple_long_short(y_true, yhat, tc_bps=tc_bps)
        signal = np.where(yhat > 0, 1.0, -1.0)
        gross = signal * y_true
        flips = (np.abs(np.diff(signal)) > 0).astype(float)
        cost = np.concatenate([[0.0], flips * (tc_bps / 1e4)])
        net = gross - cost
        cum = np.cumprod(1 + net) - 1
        fig.add_trace(go.Scatter(x=np.arange(len(cum)), y=cum, mode="lines", name=f"{name} (Sharpe {bt['sharpe']:.2f})"))
    fig.update_layout(legend=dict(orientation="h"), height=350, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")
def to_index_from_logret(arr: np.ndarray, base: float = 100.0) -> np.ndarray:
    arr = np.asarray(arr).ravel()
    return float(base) * np.exp(np.cumsum(arr))
def auto_splits_for_small_ranges(N: int, seq: int,
                                 min_train_seq: int = 16,
                                 va_min: int = 5,
                                 te_min: int = 5,
                                 min_seq: int = 5):
    """
    Returns (seq, tr, va, te) that guarantee:
      - seq >= min_seq
      - (tr - seq) >= min_train_seq  (at least this many training sequences)
      - va >= va_min, te >= te_min
      - seq + tr + va + te <= N
    """
    # clamp seq so we have room for tr/va/te
    seq = int(max(min_seq, min(seq, max(min_seq, N // 3))))
    # ensure remaining bars can satisfy min train/val/test
    min_free = min_train_seq + va_min + te_min
    if (N - seq) < min_free:
        seq = max(min_seq, N - min_free)
    free = max(0, N - seq)
    train_seq = max(min_train_seq, int(free * 0.6))
    if train_seq > free - (va_min + te_min):
        train_seq = max(min_train_seq, free - (va_min + te_min))
    va = max(va_min, int(free * 0.2))
    te = max(te_min, free - train_seq - va)
    while train_seq + va + te > free and va > va_min: va -= 1
    while train_seq + va + te > free and te > te_min: te -= 1
    tr = seq + max(0, train_seq)
    return int(seq), int(tr), int(va), int(te)
# --- tiny-range split helper (paste once) ---
def run_window(df_aligned, y_series, cond_df, cfg, device, window_id, tr, va, te, debug=False, log_cb=None):
    def log(msg):
        if log_cb:
            log_cb(msg)
    # ---- helpers (fallbacks) ----
    def _persistence_like(y_true_len):
        return np.zeros(int(y_true_len), dtype=float)
    def _safe_eval(name, y_true, y_pred):
        try:
            return evaluate_predictions(y_true, y_pred)
        except Exception as e:
            log(f"[{name}] eval failed: {e}")
            return {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan, "DPA": np.nan}
    # ---- pipeline ----
    try:
        # slices
        y_tr_raw = y_series.values[tr[0]:tr[1]]
        y_va_raw = y_series.values[va[0]:va[1]]
        y_te_raw = y_series.values[te[0]:te[1]]
        C_tr_raw = cond_df.values[tr[0]:tr[1], :]
        C_va_raw = cond_df.values[va[0]:va[1], :]
        C_te_raw = cond_df.values[te[0]:te[1], :]
        # scale
        y_scaler, y_tr, y_va, y_te = scale_fit_transform(y_tr_raw, y_va_raw, y_te_raw, is_2d=False)
        _,        C_tr, C_va, C_te = scale_fit_transform(C_tr_raw, C_va_raw, C_te_raw, is_2d=True)
        # sequences
        seq_len = cfg['data']['seq_len']
        (X_tr, Cseq_tr, t_tr) = build_sequences(y_tr, C_tr, seq_len)
        (X_va, Cseq_va, t_va) = build_sequences(np.concatenate([y_tr[-seq_len:], y_va]),
                                               np.concatenate([C_tr[-seq_len:], C_va], axis=0), seq_len)
        (X_te, Cseq_te, t_te) = build_sequences(np.concatenate([y_tr[-seq_len:], y_va, y_te]),
                                               np.concatenate([C_tr[-seq_len:], C_va, C_te], axis=0), seq_len)
        test_len = len(y_te_raw)
        X_te = X_te[-test_len:]; Cseq_te = Cseq_te[-test_len:]; t_te = t_te[-test_len:]
        y_true = y_te_raw.ravel()
        # ---- ARIMA ----
        try:
            ar_preds = []
            arr = y_tr_raw.copy()
            for i in range(len(y_te_raw)):
                yhat = arima_forecast(arr, order=(3,1,2))
                ar_preds.append(yhat)
                arr = np.concatenate([arr, [y_te_raw[i]]])
            ar_preds = np.array(ar_preds)
        except Exception as e:
            log(f"[ARIMA] error: {e}")
            ar_preds = _persistence_like(len(y_true))
        # ---- LSTM ----
        try:
            lstm_model, _ = train_lstm(X_tr, Cseq_tr, t_tr, X_va, Cseq_va, t_va,
                                       cfg['model']['lstm'], y_scaler, device=device)
            yhat_lstm_s = predict_deep(lstm_model, X_te, Cseq_te, device)
            yhat_lstm   = y_scaler.inverse_transform(yhat_lstm_s.reshape(-1,1)).ravel()
        except Exception as e:
            log(f"[LSTM] error: {e}")
            if debug: log(traceback.format_exc())
            yhat_lstm = _persistence_like(len(y_true))
        # ---- GAN ----
        try:
            G, D, _ = train_cwgan_gp(X_tr, Cseq_tr, t_tr, X_va, Cseq_va, t_va,
                                     cfg['model']['gan'], y_scaler, device=device)
            yhat_gan_s = predict_deep(G, X_te, Cseq_te, device)
            yhat_gan   = y_scaler.inverse_transform(yhat_gan_s.reshape(-1,1)).ravel()
        except Exception as e:
            log(f"[GAN] error: {e}")
            if debug: log(traceback.format_exc())
            yhat_gan = _persistence_like(len(y_true))
        # ---- Meta (safe) ----
        yhat_gan_filtered = yhat_gan.copy()
        try:
            if cfg['meta_labeling']['enabled'] and len(Cseq_va) >= 5:
                preds_val_s = predict_deep(G, X_va, Cseq_va, device)
                preds_val = y_scaler.inverse_transform(preds_val_s.reshape(-1,1)).ravel()
                y_true_val = y_series.values[va[0]:va[1]][-len(preds_val):]
                C_last_val = Cseq_va[:, -1, :]
                from stock_forecast.meta_labeling import (
                    build_meta_features, build_meta_labels, safe_train_meta_clf, apply_meta_clf
                )
                meta_X_val = build_meta_features(C_last_val, preds_val, returns_vol=None)
                meta_y_val = build_meta_labels(y_true_val, preds_val, abs_threshold=cfg['meta_labeling']['threshold_abs_pred'])
                clf = safe_train_meta_clf(meta_X_val, meta_y_val, cfg['meta_labeling']['lgbm_params'])
                C_last_te = Cseq_te[:, -1, :]
                meta_X_te = build_meta_features(C_last_te, yhat_gan, returns_vol=None)
                yhat_gan_filtered, _ = apply_meta_clf(clf, meta_X_te, yhat_gan)
        except Exception as e:
            log(f"[META] error: {e}")
            if debug: log(traceback.format_exc())
        # ---- Residual (safe) ----
        yhat_res = None
        try:
            if cfg['model']['residual']['enabled'] and len(t_tr) >= 20:
                import lightgbm as lgb
                params = cfg['model']['residual']['lgbm_params']
                F_tr = np.concatenate([X_tr[:, -1, :], Cseq_tr[:, -1, :]], axis=1)
                F_va = np.concatenate([X_va[:, -1, :], Cseq_va[:, -1, :]], axis=1)
                F_te = np.concatenate([X_te[:, -1, :], Cseq_te[:, -1, :]], axis=1)
                if len(F_tr) >= 10:
                    lgbm = lgb.LGBMRegressor(**params, random_state=42)
                    lgbm.fit(F_tr, t_tr.ravel(), eval_set=[(F_va, t_va.ravel())], verbose=False)
                    yhat_lgbm_te = lgbm.predict(F_te).reshape(-1,1)
                    yhat_lgbm_tr = lgbm.predict(F_tr).reshape(-1,1)
                    res_tr = t_tr - yhat_lgbm_tr
                    G_res, _, _ = train_cwgan_gp(X_tr, Cseq_tr, res_tr, X_va, Cseq_va, t_va,
                                                 cfg['model']['gan'], y_scaler, device=device)
                    res_hat_te_s = predict_deep(G_res, X_te, Cseq_te, device).reshape(-1,1)
                    yhat_res_s = yhat_lgbm_te + res_hat_te_s
                    yhat_res   = y_scaler.inverse_transform(yhat_res_s).ravel()
        except Exception as e:
            log(f"[RESIDUAL] error: {e}")
            if debug: log(traceback.format_exc())
            yhat_res = None
        # ---- Evaluate (never crash) ----
        eval_ar    = _safe_eval("ARIMA", y_true, ar_preds)
        eval_lstm  = _safe_eval("LSTM",  y_true, yhat_lstm)
        eval_gan   = _safe_eval("GAN",   y_true, yhat_gan)
        eval_gan_f = _safe_eval("GAN_F", y_true, yhat_gan_filtered)
        eval_res   = _safe_eval("RESID", y_true, yhat_res) if yhat_res is not None else {}
        # ---- Results ----
        results = {
            "ARIMA": eval_ar,
            "LSTM_COND": eval_lstm,
            "CWGAN_GP_COND": eval_gan,
            "CWGAN_GP_COND_META": eval_gan_f,
        }
        if yhat_res is not None:
            results["RESIDUAL_FUSION"] = eval_res
        preds = {
            "dates": y_series.index[te[0]:te[1]],
            "y_true": y_true,
            "y_arima": ar_preds,
            "y_lstm": yhat_lstm,
            "y_gan": yhat_gan,
            "y_gan_filtered": yhat_gan_filtered,
            "y_res": yhat_res
        }
        return results, preds
    except Exception as e:
        # last-resort fallback: return zeros so the GUI has something to plot
        if debug:
            log(f"[WINDOW {window_id}] FATAL: {e}")
            log(traceback.format_exc())
        y_true = y_series.values[te[0]:te[1]].ravel() if (te[1] > te[0]) else np.zeros(0, dtype=float)
        zeros = _persistence_like(len(y_true))
        results = {
            "ARIMA": _safe_eval("ARIMA", y_true, zeros),
            "LSTM_COND": _safe_eval("LSTM", y_true, zeros),
            "CWGAN_GP_COND": _safe_eval("GAN", y_true, zeros),
            "CWGAN_GP_COND_META": _safe_eval("GAN_F", y_true, zeros)
        }
        return results, {
            "dates": y_series.index[te[0]:te[1]],
            "y_true": y_true,
            "y_arima": zeros,
            "y_lstm": zeros,
            "y_gan": zeros,
            "y_gan_filtered": zeros,
            "y_res": None
        }

# gui/streamlit_app.py
import os, sys, traceback, io, json, time
from datetime import datetime
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
from stock_forecast.meta_labeling import build_meta_features, build_meta_labels, train_meta_clf, apply_meta_clf
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
def run_window(df_aligned, y_series, cond_df, cfg, device, window_id, tr, va, te, log_cb=None):
    def log(msg): 
        if log_cb: log_cb(msg)
    try:
        y_tr_raw = y_series.values[tr[0]:tr[1]]
        y_va_raw = y_series.values[va[0]:va[1]]
        y_te_raw = y_series.values[te[0]:te[1]]
        C_tr_raw = cond_df.values[tr[0]:tr[1], :]
        C_va_raw = cond_df.values[va[0]:va[1], :]
        C_te_raw = cond_df.values[te[0]:te[1], :]
        y_scaler, y_tr, y_va, y_te = scale_fit_transform(y_tr_raw, y_va_raw, y_te_raw, is_2d=False)
        c_scaler, C_tr, C_va, C_te = scale_fit_transform(C_tr_raw, C_va_raw, C_te_raw, is_2d=True)
        seq_len = cfg['data']['seq_len']
        (X_tr, Cseq_tr, t_tr) = build_sequences(y_tr, C_tr, seq_len)
        (X_va, Cseq_va, t_va) = build_sequences(np.concatenate([y_tr[-seq_len:], y_va]),
                                               np.concatenate([C_tr[-seq_len:], C_va], axis=0), seq_len)
        (X_te, Cseq_te, t_te) = build_sequences(np.concatenate([y_tr[-seq_len:], y_va, y_te]),
                                               np.concatenate([C_tr[-seq_len:], C_va, C_te], axis=0), seq_len)
        test_len = len(y_te_raw)
        X_te = X_te[-test_len:]; Cseq_te = Cseq_te[-test_len:]; t_te = t_te[-test_len:]
        y_true = y_te_raw.ravel()
        ar_preds = []
        arr = y_tr_raw.copy()
        for i in range(len(y_te_raw)):
            yhat = arima_forecast(arr, order=(3,1,2))
            ar_preds.append(yhat)
            arr = np.concatenate([arr, [y_te_raw[i]]])
        ar_preds = np.array(ar_preds)
        lstm_model, _ = train_lstm(X_tr, Cseq_tr, t_tr, X_va, Cseq_va, t_va, cfg['model']['lstm'], y_scaler, device=device)
        yhat_lstm_s = predict_deep(lstm_model, X_te, Cseq_te, device)
        yhat_lstm = y_scaler.inverse_transform(yhat_lstm_s.reshape(-1,1)).ravel()
        G, D, _ = train_cwgan_gp(X_tr, Cseq_tr, t_tr, X_va, Cseq_va, t_va, cfg['model']['gan'], y_scaler, device=device)
        yhat_gan_s = predict_deep(G, X_te, Cseq_te, device)
        yhat_gan = y_scaler.inverse_transform(yhat_gan_s.reshape(-1,1)).ravel()
        yhat_gan_filtered = yhat_gan.copy()
        if cfg['meta_labeling']['enabled']:
            preds_val_s = predict_deep(G, X_va, Cseq_va, device)
            preds_val = y_scaler.inverse_transform(preds_val_s.reshape(-1,1)).ravel()
            y_true_val = y_series.values[va[0]:va[1]][-len(preds_val):]
            C_last_val = Cseq_va[:, -1, :]
            meta_X_val = build_meta_features(C_last_val, preds_val, returns_vol=None)
            meta_y_val = build_meta_labels(y_true_val, preds_val, abs_threshold=cfg['meta_labeling']['threshold_abs_pred'])
            clf = train_meta_clf(meta_X_val, meta_y_val, cfg['meta_labeling']['lgbm_params'])
            C_last_te = Cseq_te[:, -1, :]
            meta_X_te = build_meta_features(C_last_te, yhat_gan, returns_vol=None)
            yhat_gan_filtered, _ = apply_meta_clf(clf, meta_X_te, yhat_gan)
        yhat_res = None
        if cfg['model']['residual']['enabled']:
            import lightgbm as lgb
            params = cfg['model']['residual']['lgbm_params']
            F_tr = get_cond_2d(X_tr, Cseq_tr); F_va = get_cond_2d(X_va, Cseq_va); F_te = get_cond_2d(X_te, Cseq_te)
            lgbm = lgb.LGBMRegressor(**params, random_state=42)
            lgbm.fit(F_tr, t_tr.ravel(), eval_set=[(F_va, t_va.ravel())], verbose=False)
            yhat_lgbm_te = lgbm.predict(F_te).reshape(-1,1)
            yhat_lgbm_tr = lgbm.predict(F_tr).reshape(-1,1)
            res_tr = t_tr - yhat_lgbm_tr
            G_res, _, _ = train_cwgan_gp(X_tr, Cseq_tr, res_tr, X_va, Cseq_va, t_va, cfg['model']['gan'], y_scaler, device=device)
            res_hat_te_s = predict_deep(G_res, X_te, Cseq_te, device).reshape(-1,1)
            yhat_res_s = yhat_lgbm_te + res_hat_te_s
            yhat_res = y_scaler.inverse_transform(yhat_res_s).ravel()
        eval_ar    = evaluate_predictions(y_true, ar_preds)
        eval_lstm  = evaluate_predictions(y_true, yhat_lstm)
        eval_gan   = evaluate_predictions(y_true, yhat_gan)
        eval_gan_f = evaluate_predictions(y_true, yhat_gan_filtered)
        eval_res   = evaluate_predictions(y_true, yhat_res) if yhat_res is not None else {}
        tc = cfg['backtest']['tc_bps']
        _ = simple_long_short(y_true, ar_preds, tc_bps=tc); _ = simple_long_short(y_true, yhat_lstm, tc_bps=tc)
        _ = simple_long_short(y_true, yhat_gan, tc_bps=tc); _ = simple_long_short(y_true, yhat_gan_filtered, tc_bps=tc)
        if yhat_res is not None: _ = simple_long_short(y_true, yhat_res, tc_bps=tc)
        dm_gan_vs_lstm = diebold_mariano(y_true - yhat_lstm, y_true - yhat_gan)
        dm_gan_vs_arima= diebold_mariano(y_true - ar_preds,   y_true - yhat_gan)
        results = {
            "ARIMA": {**eval_ar}, "LSTM_COND": {**eval_lstm},
            "CWGAN_GP_COND": {**eval_gan}, "CWGAN_GP_COND_META": {**eval_gan_f}
        }
        if yhat_res is not None: results["RESIDUAL_FUSION"] = {**eval_res}
        return results, {
            "dates": y_series.index[te[0]:te[1]],
            "y_true": y_true, "y_arima": ar_preds, "y_lstm": yhat_lstm,
            "y_gan": yhat_gan, "y_gan_filtered": yhat_gan_filtered, "y_res": yhat_res
        }
    except Exception as e:
        if log_cb: log_cb(f"[Window {window_id}] Error: {e}\n{traceback.format_exc()}")
        return None, None
with st.sidebar:
    st.title("Forecast Lab")
    with st.expander("Data", expanded=True):
        ticker = st.text_input("Ticker (Yahoo)", value="^GSPC")
        start  = st.date_input("Start", value=datetime(2015,1,1)).isoformat()
        end    = st.date_input("End", value=datetime.now().date()).isoformat()
        seq_len= st.number_input("Lookback (seq_len)", min_value=20, max_value=240, value=60, step=5)
    with st.expander("Features", expanded=False):
        feat_mode = st.selectbox("Feature Mode", ["real", "dummy"], index=0)
        news_csv  = st.text_input("Headlines CSV (Date, Title)", value="features/news_headlines.csv")
        aspects   = st.multiselect("Aspects (deg)", options=[0,60,90,120,180], default=[0,60,90,120,180])
        orb_deg   = st.slider("Aspect orb (deg)", min_value=1.0, max_value=5.0, value=3.0, step=0.5)
        sp_thresh = st.slider("SinglePass sim threshold", min_value=0.5, max_value=0.95, value=0.72, step=0.01)
    with st.expander("Models", expanded=False):
        enable_res = st.checkbox("Residual Fusion", value=True)
        enable_meta= st.checkbox("Meta-Labeling", value=True)
    with st.expander("Backtest", expanded=False):
        tc_bps = st.slider("Transaction cost (bps)", min_value=0.0, max_value=20.0, value=3.0, step=0.5)
    with st.expander("🕸️ Gann/Fan Overlay", expanded=False):
        enable_gann = st.checkbox("Enable overlay", value=False)
        g_pA = st.selectbox("Planet A", PLANETS, index=2, key="g_pA")
        g_pB = st.selectbox("Planet B", PLANETS, index=4, key="g_pB")
        g_aspects = st.multiselect("Aspects (deg)", [0,30,45,60,90,120,135,144,150,180], default=[0,60,90,120,180], key="g_aspects")
        g_orb = st.slider("Orb (deg)", 0.5, 5.0, 2.0, 0.5, key="g_orb")
        g_max_anchors = st.number_input("Max anchors", 1, 200, 36, 1, key="g_max")
        g_extend = st.number_input("Extend days", 0, 365, 180, 10, key="g_ext")
        g_price_step = st.number_input("Horizontal step (price units)", value=72.0, step=1.0, key="g_step")
        g_slope_scale = st.slider("1x1 slope scale", 0.10, 3.0, 1.2, 0.05, key="g_scale")
        g_both_dirs = st.checkbox("Fans both directions", True, key="g_both")
        g_add_verticals = st.checkbox("Verticals at alignments", True, key="g_v")
        g_ratios_labels = ["1x8","1x4","1x3","1x2","1x1","2x1","3x1","4x1","8x1"]
        g_ratios = st.multiselect("Fan ratios", g_ratios_labels, default=g_ratios_labels, key="g_ratios")
        st.markdown("— Planetary price lines —")
        g_show_astro = st.checkbox("Add planetary price lines", value=True, key="g_show_astro")
        g_planets = st.multiselect("Planets (lines)", PLANETS, default=["MARS","JUPITER","SATURN"], key="g_planets")
        g_harmonics = st.multiselect("Harmonics", [-2,-1,0,1,2], default=[-1,0,1], key="g_harm")
        g_deg_to_price = st.number_input("Degree → Price scale", value=3.0, step=0.1, key="g_scale_deg")
        if st.button("Clear astro cache", key="clear_astro"):
            _cached_planets.clear()
    st.markdown("---")
    build_btn = st.button("Build Features Cache")
    run_btn   = st.button("Start")
st.title("Market Forecast Lab")
st.caption(f"Device: {device_info()}")
cfg = {
    "seed": 42,
    "data": {"ticker": ticker, "start": start, "end": end, "seq_len": int(seq_len), "target": "log_return"},
    "features": {"mode": feat_mode, "news_csv_path": news_csv, "news_date_col": "Date", "news_text_col": "Title",
                 "singlepass_threshold": float(sp_thresh), "aspect_orb_deg": float(orb_deg), "aspects_deg": aspects,
                 "cache_path": "artifacts/cond_features_cache.csv"},
    "splits": {"train_len": 756, "val_len": 126, "test_len": 126, "step": 126},
    "model": {
        "lstm": {"hidden":64,"layers":2,"dropout":0.1,"epochs":40,"batch_size":64,"lr":1.0e-3,"early_stopping":8,"val_metric":"DPA"},
        "gan": {"hidden":64,"layers":2,"dropout":0.1,"epochs":80,"batch_size":64,"lr_g":2.0e-4,"lr_d":2.0e-4,"betas_g":[0.5,0.9],"betas_d":[0.5,0.9],"crit_steps":5,"lambda_gp":10.0,"early_stopping":10,"val_metric":"DPA"},
        "residual": {"enabled": bool(enable_res), "lgbm_params":{"n_estimators":400,"max_depth":4,"learning_rate":0.05,"subsample":0.9,"colsample_bytree":0.9}}
    },
    "meta_labeling": {"enabled": bool(enable_meta), "threshold_abs_pred": 0.0, "lgbm_params":{"n_estimators":250,"max_depth":3,"learning_rate":0.05,"subsample":0.9,"colsample_bytree":0.9}},
    "ablation": {"residual_compare": True, "meta_compare": True},
    "backtest": {"tc_bps": float(tc_bps)}
}
if build_btn and feat_mode == "real":
    ensure_dir("artifacts")
    st.write("Building features (astro + event)…")
    df_tmp = fetch_ohlc_yf(ticker, start=start, end=end)
    cond_df, cond_dim = assemble_conditional("real", df_tmp, cfg["features"])
    st.success(f"Cached at {cfg['features']['cache_path']} (cond_dim={cond_dim}).")
if run_btn:
    set_seed(cfg.get("seed", 42))
    import torch; device = "cuda" if torch.cuda.is_available() else "cpu"
    df = fetch_ohlc_yf(ticker, start=start, end=end)
    y_series = make_target(df['Close'], mode="log_return")
    cond_df, cond_dim = assemble_conditional(cfg['features']['mode'], df, cfg['features'])
    idx = y_series.index.intersection(cond_df.index)
    y_series = y_series.loc[idx]; cond_df = cond_df.loc[idx]; df_aligned = df.loc[idx]
    N = len(y_series); seq = int(cfg['data']['seq_len'])
    tr = cfg['splits']['train_len']; va = cfg['splits']['val_len']; te = cfg['splits']['test_len']; step = cfg['splits']['step']
    min_need = seq + tr + va + te
    if N <= min_need:
        free = max(N - seq, 30)
        tr = max(int(free*0.6), 30); va = max(int(free*0.2), 10); te = max(free - tr - va, 10); step = max(te, 10)
    st.markdown(f'<div class="run-header">Ticker {ticker} | {start} → {end} | N={N} | splits (tr={tr}, va={va}, te={te})</div>', unsafe_allow_html=True)
    progress_bar = st.progress(0, text="Preparing…")
    results_all, first_preds = [], None
    total_windows = sum(1 for _ in rolling_windows(N, seq, tr, va, te, step))
    if total_windows == 0:
        st.error("Not enough data for one window. Widen date range or reduce lookback.")
        st.stop()
    for w_id, (tr_i, va_i, te_i) in enumerate(rolling_windows(N, seq, tr, va, te, step), start=1):
        progress_bar.progress(int((w_id / total_windows) * 100), text=f"Window {w_id}/{total_windows}")
        wr, preds = run_window(df_aligned, y_series, cond_df, cfg, device, w_id, tr_i, va_i, te_i)
        if wr is None: continue
        wr["window"] = {"start": str(df_aligned.index[te_i[0]]), "end": str(df_aligned.index[te_i[1]-1])}
        results_all.append(wr)
        if first_preds is None: first_preds = preds
    st.success(f"Completed {len(results_all)} windows.")
    tabs = st.tabs(["Dashboard", "Predictions", "Gann/Fan Grid", "Features", "Windows", "Logs"])
    with tabs[0]:
        st.subheader("Averaged Performance")
        def aggregate_with_ci(results_all, model_key):
            keys = set()
            for r in results_all:
                if model_key in r: keys.update(r[model_key].keys())
            stats = {}
            for k in keys:
                vals = [r[model_key][k] for r in results_all if model_key in r and k in r[model_key]]
                if len(vals) == 0: continue
                stats[k] = {"mean": float(np.mean(vals))}
            return stats
        model_blocks = {"ARIMA":"ARIMA","LSTM (Cond)":"LSTM_COND","cWGAN-GP (Cond)":"CWGAN_GP_COND","cWGAN-GP + Meta":"CWGAN_GP_COND_META","Residual Fusion":"RESIDUAL_FUSION"}
        cols = st.columns(len(model_blocks))
        for i, (title, key) in enumerate(model_blocks.items()):
            stats = aggregate_with_ci(results_all, key)
            dpa = stats.get("DPA", {}).get("mean", None); rmse = stats.get("RMSE", {}).get("mean", None)
            with cols[i]:
                kpi_card(title, f"DPA {dpa:.3f}" if dpa is not None else "N/A")
                st.caption(f"RMSE {rmse:.4f}" if rmse is not None else "")
        st.markdown("---")
        st.subheader("Equity Curves (First Test Window)")
        if first_preds:
            curves = {"ARIMA": first_preds["y_arima"], "LSTM": first_preds["y_lstm"], "GAN": first_preds["y_gan"], "GAN (Meta)": first_preds["y_gan_filtered"]}
            if first_preds["y_res"] is not None: curves["Residual"] = first_preds["y_res"]
            plot_equity_curve(first_preds["y_true"], curves, tc_bps=cfg["backtest"]["tc_bps"])
        else:
            st.info("No predictions captured.")
    with tabs[1]:
        st.subheader("Predictions vs True — First Test Window")
        if first_preds:
            df_plot = pd.DataFrame({"Date": first_preds["dates"], "True": first_preds["y_true"],
                                    "ARIMA": first_preds["y_arima"], "LSTM": first_preds["y_lstm"],
                                    "GAN": first_preds["y_gan"], "GAN (Meta)": first_preds["y_gan_filtered"]})
            if first_preds["y_res"] is not None: df_plot["Residual"] = first_preds["y_res"]
            model_cols = [c for c in df_plot.columns if c not in ["Date","True"]]
            use_index = st.checkbox("Convert to cumulative index (base=100)", value=True)
            if use_index:
                df_plot["True"] = to_index_from_logret(df_plot["True"].values)
                for c in model_cols: df_plot[c] = to_index_from_logret(df_plot[c].values)
            use_holdings = st.checkbox("Show as Firm Holdings (stacked area)", value=True)
            if use_holdings: plot_firm_holdings(df_plot, model_cols)
            else: plot_predictions(df_plot, model_cols)
            st.download_button("Download first-window predictions CSV", data=df_plot.to_csv(index=False).encode("utf-8"), file_name="preds_first_window.csv", mime="text/csv")
        else:
            st.info("No predictions to show.")
    with tabs[2]:
        st.subheader("Price with Gann/Fan + Planetary Overlay")
        dates = df_aligned.index; close = df_aligned["Close"]
        if enable_gann:
            ratios_map = {"1x8":1/8,"1x4":1/4,"1x3":1/3,"1x2":1/2,"1x1":1,"2x1":2,"3x1":3,"4x1":4,"8x1":8}
            shapes = build_overlay_shapes(dates, close, (g_pA, g_pB), g_aspects, float(g_orb), int(g_max_anchors),
                                          [ratios_map[r] for r in g_ratios], float(g_slope_scale), int(g_extend),
                                          bool(g_both_dirs), bool(g_add_verticals), float(g_price_step))
            end_ext = dates[-1] + pd.Timedelta(days=int(g_extend))
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=dates, open=df_aligned["Open"], high=df_aligned["High"], low=df_aligned["Low"], close=close,
                                         increasing_line_color="#2ECC71", decreasing_line_color="#E74C3C"))
            if g_show_astro and g_planets:
                for tr in planet_price_traces(dates, close, g_planets, g_harmonics, g_deg_to_price):
                    fig.add_trace(tr)
            fig.update_layout(height=760, margin=dict(l=0,r=0,t=24,b=0), xaxis=dict(range=[dates[0], end_ext]), shapes=shapes, showlegend=g_show_astro)
            st.plotly_chart(fig, width="stretch")
            html = fig.to_html(full_html=False, include_plotlyjs="cdn")
            st.download_button("Download overlay (HTML)", data=html, file_name="gann_overlay.html", mime="text/html")
        else:
            st.info("Enable overlay in the sidebar.")
    with tabs[3]:
        st.subheader("Feature Explorer")
        cache_path = cfg['features']['cache_path']
        if os.path.exists(cache_path):
            cdf = pd.read_csv(cache_path, parse_dates=['Date']).set_index('Date')
            st.write("Columns:", list(cdf.columns))
            sel_cols = st.multiselect("Select features to plot", options=list(cdf.columns), default=list(cdf.columns)[:4])
            if sel_cols:
                fig = go.Figure()
                for col in sel_cols: fig.add_trace(go.Scatter(x=cdf.index, y=cdf[col], mode="lines", name=col))
                fig.update_layout(legend=dict(orientation="h"), height=400, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig, width="stretch")
            st.download_button("Download cached features CSV", data=cdf.to_csv().encode("utf-8"), file_name="cond_features_cache.csv")
        else:
            st.warning("No cached features found. Click 'Build Features Cache' first (Real Mode).")
    with tabs[4]:
        st.subheader("Windows")
        if results_all:
            rows = []
            for r in results_all:
                ws = r["window"]["start"]; we = r["window"]["end"]
                for label, key in [("ARIMA","ARIMA"),("LSTM_COND","LSTM_COND"),("CWGAN_GP_COND","CWGAN_GP_COND"),("CWGAN_GP_COND_META","CWGAN_GP_COND_META"),("RESIDUAL_FUSION","RESIDUAL_FUSION")]:
                    if key in r:
                        row = {"WindowStart": ws, "WindowEnd": we, "Model": label}; row.update(r[key]); rows.append(row)
            if rows: st.dataframe(pd.DataFrame(rows))
            else: st.info("No window results found.")
        else:
            st.info("No runs yet.")
    with tabs[5]:
        st.subheader("Logs")
        st.caption("See the progress banner and tabs for details.")

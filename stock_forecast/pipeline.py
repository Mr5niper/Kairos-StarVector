"""
stock_forecast.pipeline
=======================
Conditioning features and the walk-forward benchmark, wired to the new
`kairos` engine.

What the conditioning features are now
--------------------------------------
The old build fed the models three hand-rolled astro columns:
`astro_csw`, `astro_bradley` and `astro_gann_prox`. The first two were
reasonable ideas, but `astro_gann_prox` was closeness to a round-number
price grid, computed from the closing price of the same bar the model was
predicting. That is lookahead: the feature contained the answer. Any
apparent skill it produced was leakage, not signal.

It is gone. The feature set is now:

  vol_atr, ma_ratio, returns_lag1   price context, all lagged
  astro_wave                        the trickle-down composite
  harmonic_1, harmonic_2, harmonic_4  continuous alignment indices
  event_heat                        weighted alignment density
  event_sentiment                   news sentiment, only if the optional
                                    transformers extras are installed

Every column is either lagged or derived purely from the ephemeris, which
is knowable in advance. Nothing reads the bar being predicted.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from kairos import astro as A
from kairos import waves as W

DEFAULT_BODIES = ["SUN", "VENUS", "MARS", "JUPITER", "SATURN", "URANUS", "NEPTUNE"]
DEFAULT_ASPECTS = [0.0, 90.0, 120.0, 180.0]


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def build_conditional_features(
    price_df: pd.DataFrame,
    *,
    bodies: Sequence[str] = tuple(DEFAULT_BODIES),
    aspects: Sequence[float] = tuple(DEFAULT_ASPECTS),
    orb_deg: float = 2.0,
    tau_days: float = 30.0,
    horizon_days: int = 180,
    harmonics: Sequence[int] = (1, 2, 4),
    wave: Optional[pd.Series] = None,
    lons: Optional[pd.DataFrame] = None,
    news_csv: Optional[str] = None,
    frame: str = "geocentric",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Assemble the conditioning matrix aligned to the price index.

    Pass `wave` and `lons` to reuse what the GUI already computed instead
    of recalculating the ephemeris.
    """
    idx = pd.DatetimeIndex(price_df.index)
    first, last = idx.min(), idx.max()

    if lons is None:
        cal = A.calendar_index(first, last)
        lons = A.longitudes(cal, bodies=list(bodies), frame=frame)

    if wave is None:
        events = A.aspect_events(
            lons, A.EventSpec(aspects=list(aspects), orb_deg=float(orb_deg))
        )
        wave = W.composite_pressure(lons.index, events,
                                    tau_days=tau_days, horizon_days=horizon_days)
    else:
        events = A.aspect_events(
            lons, A.EventSpec(aspects=list(aspects), orb_deg=float(orb_deg))
        )

    feat = pd.DataFrame(index=idx)

    # Price context. Every column is shifted so no bar sees itself.
    high, low, close = price_df["High"], price_df["Low"], price_df["Close"]
    feat["vol_atr"] = (high - low).rolling(14).mean().shift(1)
    feat["ma_ratio"] = (close / close.rolling(20).mean()).shift(1)
    feat["returns_lag1"] = close.pct_change().shift(1)

    # Astro. Deterministic and known ahead of time, so no shift needed.
    feat["astro_wave"] = wave.reindex(idx).ffill()
    for h in harmonics:
        feat[f"harmonic_{h}"] = A.harmonic_index(lons, harmonic=int(h)).reindex(idx).ffill()

    if events is not None and not events.empty:
        heat = (events.assign(d=pd.to_datetime(events["date"]).dt.normalize())
                      .groupby("d")["weight"].sum())
        feat["event_heat"] = heat.reindex(idx).fillna(0.0).rolling(21).sum()
    else:
        feat["event_heat"] = 0.0

    if news_csv and os.path.exists(news_csv):
        sent = _news_sentiment(idx, news_csv)
        if sent is not None:
            feat["event_sentiment"] = sent

    feat = feat.replace([np.inf, -np.inf], np.nan).dropna()
    return feat, list(feat.columns)


def _news_sentiment(idx: pd.DatetimeIndex, news_csv: str) -> Optional[pd.Series]:
    """
    Daily FinBERT sentiment, if the optional extras are installed.

    Batched across the whole corpus rather than looping day by day as the
    old code did. That version re-entered the transformers pipeline once
    per calendar date, which on ten years of data meant several thousand
    separate calls and a runtime measured in hours.
    """
    try:
        from transformers import pipeline as hf_pipeline
    except Exception:
        return None
    try:
        raw = pd.read_csv(news_csv)
        date_col = next((c for c in raw.columns if c.lower() in ("date", "day")), None)
        text_col = next((c for c in raw.columns
                         if c.lower() in ("title", "headline", "text")), None)
        if not date_col or not text_col:
            return None
        raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce").dt.normalize()
        raw = raw.dropna(subset=[date_col, text_col])
        if raw.empty:
            return None

        # top_k=None replaces return_all_scores, which was removed in
        # transformers 5. Passing the old kwarg raises rather than warns.
        clf = hf_pipeline("text-classification", model="ProsusAI/finbert",
                          top_k=None, truncation=True)
        texts = raw[text_col].astype(str).tolist()
        scored = clf(texts, batch_size=32)

        signed = []
        for row in scored:
            m = {d["label"].lower(): float(d["score"]) for d in row}
            signed.append(m.get("positive", 0.0) - m.get("negative", 0.0))
        raw = raw.assign(_s=signed)
        daily = raw.groupby(date_col)["_s"].mean()
        return daily.reindex(idx).ffill().fillna(0.0)
    except Exception:
        return None


# --------------------------------------------------------------------------
# Walk-forward benchmark
# --------------------------------------------------------------------------
def rolling_windows(
    n: int, seq_len: int, train_len: int, val_len: int, test_len: int, step: int
):
    """Expanding train, fixed validation and test, stepping forward."""
    start = train_len
    while start + val_len + test_len <= n:
        yield (0, start), (start, start + val_len), (start + val_len, start + val_len + test_len)
        start += step


def auto_split_sizes(n: int, seq_len: int) -> Tuple[int, int, int, int]:
    """Delegates to splits.fit_split_sizes so there is one implementation."""
    from .splits import fit_split_sizes
    return fit_split_sizes(n, seq_len)


def _scale(train: np.ndarray, *others: np.ndarray, is_2d: bool = False):
    """Standardise on train statistics only. No sklearn dependency needed."""
    tr = train.reshape(-1, 1) if not is_2d else train
    mu = np.nanmean(tr, axis=0)
    sd = np.nanstd(tr, axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)

    class Scaler:
        def transform(self, x):
            a = x.reshape(-1, 1) if not is_2d else x
            return ((a - mu) / sd).astype(np.float32)

        def inverse_transform(self, x):
            a = np.asarray(x, dtype=float)
            if not is_2d and a.ndim == 1:
                a = a.reshape(-1, 1)
            return a * sd + mu

    sc = Scaler()
    out = [sc.transform(tr)]
    for o in others:
        out.append(sc.transform(o))
    if not is_2d:
        out = [o.ravel() for o in out]
    return (sc, *out)


def build_sequences(y: np.ndarray, c: np.ndarray, seq_len: int):
    n = len(y)
    if n <= seq_len:
        return (np.zeros((0, seq_len, 1), np.float32),
                np.zeros((0, seq_len, c.shape[1]), np.float32),
                np.zeros((0, 1), np.float32))
    xs = np.stack([y[i - seq_len:i] for i in range(seq_len, n)]).astype(np.float32)
    cs = np.stack([c[i - seq_len:i, :] for i in range(seq_len, n)]).astype(np.float32)
    ts = y[seq_len:n].astype(np.float32).reshape(-1, 1)
    return xs.reshape(len(xs), seq_len, 1), cs, ts


def run_benchmark_gui(
    df: pd.DataFrame,
    wave: Optional[pd.Series] = None,
    lons: Optional[pd.DataFrame] = None,
    seq_len: int = 60,
    epochs: int = 25,
    max_windows: int = 2,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, object]:
    """
    Walk-forward comparison of ARIMA, a conditional LSTM and a conditional
    WGAN-GP, using the alignment features as conditioning.

    Returns a summary table and the first test window's predictions.
    """
    def say(msg: str) -> None:
        if log:
            log(msg)

    import torch

    from .metrics import directional_accuracy, mae, rmse
    from .models.arima import arima_forecast
    from .train_gan import train_cwgan_gp
    from .train_lstm import train_lstm

    cond, cols = build_conditional_features(df, wave=wave, lons=lons)
    close = df["Close"]
    y = np.log(close).diff().dropna()

    idx = y.index.intersection(cond.index)
    if len(idx) < 120:
        raise RuntimeError(
            f"Only {len(idx)} aligned bars after feature construction; "
            "need at least 120. Widen the date range."
        )
    y = y.loc[idx]
    cond = cond.loc[idx]
    say(f"Aligned {len(idx)} bars with {len(cols)} conditioning features.")

    n = len(y)
    seq, tr_len, va_len, te_len = auto_split_sizes(n, seq_len)
    step = max(te_len, 10)
    say(f"seq={seq} train={tr_len} val={va_len} test={te_len}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    say(f"Device: {device}")

    lstm_cfg = {"hidden": 64, "layers": 2, "dropout": 0.1, "epochs": int(epochs),
                "batch_size": 64, "lr": 1e-3, "early_stopping": 6, "val_metric": "DPA"}
    gan_cfg = {"hidden": 64, "layers": 2, "dropout": 0.1, "epochs": int(epochs),
               "batch_size": 64, "lr_g": 2e-4, "lr_d": 2e-4,
               "betas_g": [0.5, 0.9], "betas_d": [0.5, 0.9],
               "crit_steps": 3, "lambda_gp": 10.0,
               "early_stopping": 8, "val_metric": "DPA"}

    windows = list(rolling_windows(n, seq, tr_len, va_len, te_len, step))
    if max_windows and max_windows > 0:
        windows = windows[:int(max_windows)]
    if not windows:
        raise RuntimeError("No walk-forward window fits this date range. Widen it.")
    say(f"{len(windows)} window(s).")

    rows: List[Dict] = []
    first_preds: Optional[pd.DataFrame] = None

    for w_i, (tr, va, te) in enumerate(windows, start=1):
        say(f"Window {w_i}/{len(windows)}...")
        y_tr, y_va, y_te = (y.values[tr[0]:tr[1]], y.values[va[0]:va[1]],
                            y.values[te[0]:te[1]])
        c_tr, c_va, c_te = (cond.values[tr[0]:tr[1]], cond.values[va[0]:va[1]],
                            cond.values[te[0]:te[1]])

        ysc, ys_tr, ys_va, ys_te = _scale(y_tr, y_va, y_te, is_2d=False)
        _, cs_tr, cs_va, cs_te = _scale(c_tr, c_va, c_te, is_2d=True)

        X_tr, C_tr, T_tr = build_sequences(ys_tr, cs_tr, seq)
        X_va, C_va, T_va = build_sequences(
            np.concatenate([ys_tr[-seq:], ys_va]),
            np.concatenate([cs_tr[-seq:], cs_va], axis=0), seq)
        X_te, C_te, T_te = build_sequences(
            np.concatenate([ys_tr[-seq:], ys_va, ys_te]),
            np.concatenate([cs_tr[-seq:], cs_va, cs_te], axis=0), seq)
        keep = len(y_te)
        X_te, C_te = X_te[-keep:], C_te[-keep:]
        truth = y_te.ravel()

        preds: Dict[str, np.ndarray] = {}

        try:
            acc, arr = [], y_tr.copy()
            for i in range(keep):
                acc.append(arima_forecast(arr, order=(2, 0, 1)))
                arr = np.concatenate([arr, [y_te[i]]])
            preds["ARIMA"] = np.asarray(acc, dtype=float)
        except Exception as exc:
            say(f"  ARIMA failed: {exc}")

        def predict(model) -> np.ndarray:
            model.eval()
            with torch.no_grad():
                out = model(torch.from_numpy(X_te).to(device),
                            torch.from_numpy(C_te).to(device))
            return ysc.inverse_transform(out.cpu().numpy().ravel()).ravel()

        try:
            m, _ = train_lstm(X_tr, C_tr, T_tr, X_va, C_va, T_va,
                              lstm_cfg, ysc, device=device)
            preds["LSTM"] = predict(m)
        except Exception as exc:
            say(f"  LSTM failed: {exc}")

        try:
            from .train_gan import predict_mean
            g, _, _ = train_cwgan_gp(X_tr, C_tr, T_tr, X_va, C_va, T_va,
                                     gan_cfg, ysc, device=device)
            # Averaged over noise draws: a single pass would be one sample
            # from the generator's distribution, not its expectation.
            preds["cWGAN-GP"] = ysc.inverse_transform(
                predict_mean(g, X_te, C_te, device, n_samples=32)).ravel()
        except Exception as exc:
            say(f"  GAN failed: {exc}")

        # Persistence: the honest floor. Any model that cannot beat "assume
        # tomorrow looks like today" has learned nothing worth having.
        preds["Persistence"] = np.full(keep, float(np.mean(y_tr)))

        for name, p in preds.items():
            p = np.asarray(p, dtype=float).ravel()[:keep]
            if len(p) != keep:
                continue
            rows.append({
                "window": w_i, "model": name,
                "RMSE": round(rmse(truth, p), 6),
                "MAE": round(mae(truth, p), 6),
                "DPA": round(directional_accuracy(truth, p), 4),
            })

        if first_preds is None:
            frame = {"actual": truth}
            frame.update({k: np.asarray(v, dtype=float).ravel()[:keep]
                          for k, v in preds.items()})
            first_preds = pd.DataFrame(frame, index=y.index[te[0]:te[1]])

    detail = pd.DataFrame(rows)
    summary = (detail.groupby("model")[["RMSE", "MAE", "DPA"]]
               .mean().round(6).reset_index()
               .sort_values("DPA", ascending=False)) if not detail.empty else detail

    return {"summary": summary, "detail": detail, "predictions": first_preds,
            "features": cols}

# stock_forecast/dataset.py
# ============================================
# Dataset & Feature Engineering (FINAL PATCHED)
# - Safe, version-agnostic Skyfield ecliptic handling
# - Chunked, numpy-only timescale construction (no pandas index mutation)
# - Robust tuple unpacking (2 or 3 values) from Skyfield lat/lon
# - All arrays flattened to 1D before building DataFrames (fixes ValueError: shape (N,1))
# - Event/Sentiment pipeline (RSS CSV -> SBERT SinglePass -> FinBERT)
# - Conditional feature assembly (real/dummy)
# - Sequence building & scaling
# ============================================
import os
from typing import Tuple, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
# ---------- Skyfield imports (handle version differences for ecliptic) ----------
from skyfield.api import load
try:
    # Newer Skyfield versions
    from skyfield.framelib import ecliptic_frame
    _HAS_ECLIPTIC_FRAME = True
except Exception:
    # Older Skyfield versions: fallback path will be used
    ecliptic_frame = None
    _HAS_ECLIPTIC_FRAME = False

# ---------------------------------------------------------------------------
# Core price utilities
# ---------------------------------------------------------------------------
def fetch_ohlc_yf(ticker: str, start: str = "2000-01-01", end: Optional[str] = None) -> pd.DataFrame:
    """Fetch OHLCV from Yahoo Finance via yfinance."""
    import yfinance as yf
    data = yf.download(ticker, start=start, end=end, auto_adjust=True)
    data = data.dropna()
    data.index = pd.to_datetime(data.index)
    return data

def make_target(series: pd.Series, mode: str = "log_return") -> pd.Series:
    """Build target series: log returns or close deltas."""
    series = series.astype(float)
    if mode == "log_return":
        y = np.log(series).diff()
    elif mode == "close_delta":
        y = series.diff()
    else:
        raise ValueError(f"Unknown target mode: {mode}")
    return y.dropna()

# ---------------------------------------------------------------------------
# Astro (Skyfield) pipeline
# ---------------------------------------------------------------------------
def _planet_positions(dates: pd.DatetimeIndex, chunk_size: int = 1000) -> Dict[str, np.ndarray]:
    """
    Compute ecliptic longitudes for selected bodies for each date using Skyfield.
    FIXES:
    - Robust against pandas Index mutability by converting to numpy arrays and chunking.
    - Robust tuple unpacking: handle (lat, lon) or (lat, lon, distance).
    - Ensures 1D arrays for downstream DataFrame construction.
    """
    eph = load('de421.bsp')  # downloaded once by skyfield
    ts = load.timescale()
    planets = {
        'SUN': eph['sun'],
        'MERCURY': eph['mercury'],
        'VENUS': eph['venus'],
        'MARS': eph['mars'],
        'JUPITER': eph['jupiter barycenter'],
        'SATURN': eph['saturn barycenter'],
    }
    earth = eph['earth']
    # Convert pandas DateTimeIndex fields to plain numpy arrays (NOT pandas Index)
    years = dates.year.to_numpy(dtype=int)
    months = dates.month.to_numpy(dtype=int)
    days = dates.day.to_numpy(dtype=int)
    N = len(dates)
    longs: Dict[str, np.ndarray] = {name: np.empty(N, dtype=float) for name in planets}
    # Process in chunks
    for start in range(0, N, chunk_size):
        end = min(N, start + chunk_size)
        t = ts.utc(years[start:end], months[start:end], days[start:end])  # pure numpy --> safe
        obs = earth.at(t)
        for name, body in planets.items():
            app = obs.observe(body).apparent()
            if _HAS_ECLIPTIC_FRAME and ecliptic_frame is not None:
                tpl = app.frame_latlon(ecliptic_frame)  # (lat, lon) or (lat, lon, distance)
            else:
                tpl = app.ecliptic_latlon()            # fallback for older skyfield
            # Robust unpacking
            if isinstance(tpl, (tuple, list)):
                if len(tpl) == 3:
                    lat, lon, _dist = tpl
                elif len(tpl) == 2:
                    lat, lon = tpl
                else:
                    raise ValueError(f"Unexpected tuple length from Skyfield lat/lon: {len(tpl)}")
            else:
                # Rare case; assume it's already (lat, lon)
                lat, lon = tpl
            lon_deg = np.asarray(lon.degrees).reshape(-1) % 360.0  # ensure 1D
            longs[name][start:end] = lon_deg
    return longs

def _aspect_score(lon_a: np.ndarray, lon_b: np.ndarray, aspects: list, orb: float) -> np.ndarray:
    """
    Compute aspect score between two longitudes for a set of aspects.
    Uses a smooth cosine kernel within 'orb' degrees of each aspect angle.
    Ensures 1D output.
    """
    lon_a = np.asarray(lon_a).reshape(-1)
    lon_b = np.asarray(lon_b).reshape(-1)
    # Angular distance in [0,180]
    diff = np.abs((lon_a - lon_b + 180) % 360 - 180)
    score = np.zeros(diff.shape, dtype=float)
    for a in aspects:
        d = np.abs(diff - a)
        within = d <= orb
        # Smooth peaked contribution within orb
        contrib = np.cos((d * np.pi) / (2 * orb))
        contrib[~within] = 0.0
        score += contrib
    return score.reshape(-1)

def generate_astro_features_real(
    dates: pd.DatetimeIndex,
    price_df: pd.DataFrame,
    aspects: list,
    orb: float
) -> pd.DataFrame:
    """
    Compute daily astro scores for CSW-like, Bradley-like, and a simple Gann proximity proxy.
    Ensures all columns are 1D arrays when building DataFrame.
    """
    longs = _planet_positions(dates)
    planets = ['SUN', 'MERCURY', 'VENUS', 'MARS', 'JUPITER', 'SATURN']
    N = len(dates)
    # CSW-like: sum of pairwise aspect scores
    csw = np.zeros(N, dtype=float)
    # Bradley-like: sum of cosines of separations
    bradley = np.zeros(N, dtype=float)
    for i in range(len(planets)):
        for j in range(i + 1, len(planets)):
            csw += _aspect_score(longs[planets[i]], longs[planets[j]], aspects, orb)
            sep = np.abs((longs[planets[i]] - longs[planets[j]] + 180) % 360 - 180)
            sep = np.asarray(sep).reshape(-1)
            bradley += np.cos(np.deg2rad(sep))
    # Gann proximity proxy: closeness to round price grid
    close = price_df.loc[dates, 'Close'].values.reshape(-1)
    round_grid = np.round(close / 10.0) * 10.0
    denom = np.maximum(np.abs(close), 1e-6)
    gann_prox = 1.0 - (np.abs(close - round_grid) / denom)
    # Flatten all to 1D explicitly
    csw = np.asarray(csw).reshape(-1)
    bradley = np.asarray(bradley).reshape(-1)
    gann_prox = np.asarray(gann_prox).reshape(-1)
    astro_df = pd.DataFrame({
        "astro_csw": csw,
        "astro_bradley": bradley,
        "astro_gann_prox": gann_prox
    }, index=dates)
    astro_df = astro_df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return astro_df

# ---------------------------------------------------------------------------
# Event/Sentiment pipeline (SBERT + SinglePass + FinBERT)
# ---------------------------------------------------------------------------
def _singlepass_cluster(embeddings: np.ndarray, sim_thresh: float = 0.72) -> np.ndarray:
    """
    SinglePass clustering on cosine similarity of embeddings.
    Returns cluster labels per item (not used for scoring here, but included for completeness).
    """
    from sklearn.metrics.pairwise import cosine_similarity
    labels = -np.ones(len(embeddings), dtype=int)
    clusters = []
    for i, emb in enumerate(embeddings):
        if not clusters:
            clusters.append([i]); labels[i] = 0; continue
        best, best_sim = -1, -1.0
        for cidx, members in enumerate(clusters):
            sims = cosine_similarity(embeddings[members], emb.reshape(1, -1)).ravel()
            avg_sim = float(np.mean(sims))
            if avg_sim > best_sim:
                best_sim, best = avg_sim, cidx
        if best_sim >= sim_thresh:
            clusters[best].append(i)
            labels[i] = best
        else:
            clusters.append([i])
            labels[i] = len(clusters) - 1
    return labels

def generate_event_sentiment_real(
    dates: pd.DatetimeIndex,
    news_csv: str,
    date_col: str,
    text_col: str,
    sim_thresh: float
) -> pd.DataFrame:
    """
    Build daily event features (topic heat, negative heat, avg sentiment) using SBERT + SinglePass + FinBERT.
    Returns a DataFrame aligned to 'dates' with columns: event_heat, event_heat_neg, event_sentiment.
    If CSV or models are missing, returns zeros (no crash).
    """
    # Prepare the output frame (zeros by default)
    ev_df = pd.DataFrame(index=dates, columns=["event_heat", "event_heat_neg", "event_sentiment"], dtype=float)
    ev_df.loc[:, :] = 0.0
    #--- Load news CSV
    if not os.path.exists(news_csv):
        print(f"[Event] News CSV not found: {news_csv}. Using zero event features.")
        return ev_df
    raw = pd.read_csv(news_csv)
    if date_col not in raw.columns or text_col not in raw.columns:
        print(f"[Event] News CSV missing required columns ({date_col}, {text_col}). Using zero event features.")
        return ev_df
    # Normalize date and drop missing
    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce").dt.date
    raw = raw.dropna(subset=[date_col, text_col])
    if raw.empty:
        print("[Event] News CSV empty after parsing. Using zero event features.")
        return ev_df
    # Lazy-load models
    try:
        from sentence_transformers import SentenceTransformer
        sbert = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    except Exception as e:
        print(f"[Event] SBERT load failed: {e}. Using zero event features.")
        return ev_df
    try:
        from transformers import pipeline
        finbert_name = "ProsusAI/finbert"
        fin_clf = pipeline("text-classification", model=finbert_name, tokenizer=finbert_name, return_all_scores=True)
    except Exception as e:
        print(f"[Event] FinBERT load failed: {e}. Using zero event features.")
        return ev_df
    # Daily aggregation
    for d in dates:
        d_only = d.date()
        day_texts = raw.loc[raw[date_col] == d_only, text_col].astype(str).tolist()
        if len(day_texts) == 0:
            continue
        try:
            embs = sbert.encode(day_texts, normalize_embeddings=True)
            _ = _singlepass_cluster(np.array(embs), sim_thresh=sim_thresh)
        except Exception as e:
            print(f"[Event] SBERT embedding failed for {d_only}: {e}; continuing sentiment only.")
        try:
            sentiments = fin_clf(day_texts, truncation=True)
        except Exception as e:
            print(f"[Event] FinBERT sentiment failed for {d_only}: {e}")
            continue
        heat = len(day_texts)
        neg_heat = 0
        sent_scores = []
        for s in sentiments:
            # FinBERT returns a list of dicts [{"label": "...", "score": float}, ...]
            label_to_score = {row["label"].lower(): row["score"] for row in s}
            signed = label_to_score.get("positive", 0.0) - label_to_score.get("negative", 0.0)
            sent_scores.append(signed)
            if label_to_score.get("negative", 0.0) > 0.5:
                neg_heat += 1
        ev_df.at[d, "event_heat"] = float(heat)
        ev_df.at[d, "event_heat_neg"] = float(neg_heat)
        ev_df.at[d, "event_sentiment"] = float(np.mean(sent_scores) if sent_scores else 0.0)
    return ev_df.fillna(0.0)

# ---------------------------------------------------------------------------
# Conditional feature assembly
# ---------------------------------------------------------------------------
def assemble_conditional(mode: str, price_df: pd.DataFrame, feat_cfg: dict) -> Tuple[pd.DataFrame, int]:
    """
    Assemble conditional features in either 'real' or 'dummy' mode.
    Real mode uses: Skyfield astro + SBERT/FinBERT event/sentiment (from CSV headlines).
    Returns (cond_df, cond_dim).
    """
    if mode == "dummy":
        base = pd.DataFrame(index=price_df.index)
        base['vol_atr'] = (price_df['High'] - price_df['Low']).rolling(14).mean()
        base['ma_ratio'] = price_df['Close'] / price_df['Close'].rolling(20).mean()
        base['returns_lag1'] = price_df['Close'].pct_change().shift(1)
        # Simple placeholders (ensure 1D)
        t = np.linspace(0, 8*np.pi, len(base))
        base['astro_csw'] = np.sin(t)*0.3
        base['event_sentiment'] = np.tanh(np.random.randn(len(base))*0.7)
        base = base.dropna()
        return base, base.shape[1]
    if mode != "real":
        raise ValueError("features.mode must be 'real' or 'dummy'")
    cache_path = feat_cfg.get("cache_path", None)
    if cache_path and os.path.exists(cache_path):
        df_cached = pd.read_csv(cache_path, parse_dates=['Date']).set_index('Date')
        return df_cached, df_cached.shape[1]
    dates = price_df.index
    # Astro
    aspects = feat_cfg.get("aspects_deg", [0, 60, 90, 120, 180])
    orb = float(feat_cfg.get("aspect_orb_deg", 3.0))
    astro_df = generate_astro_features_real(dates, price_df, aspects, orb)
    # Event/Sentiment
    ev_df = generate_event_sentiment_real(
        dates,
        feat_cfg['news_csv_path'],
        feat_cfg['news_date_col'],
        feat_cfg['news_text_col'],
        feat_cfg.get('singlepass_threshold', 0.72)
    )
    # Technical context
    base = pd.DataFrame(index=dates)
    base['vol_atr'] = (price_df['High'] - price_df['Low']).rolling(14).mean()
    base['ma_ratio'] = price_df['Close'] / price_df['Close'].rolling(20).mean()
    base['returns_lag1'] = price_df['Close'].pct_change().shift(1)
    cond_df = pd.concat([base, astro_df, ev_df], axis=1).dropna()
    if cache_path:
        out = cond_df.copy()
        out.insert(0, 'Date', out.index)
        out.to_csv(cache_path, index=False)
    return cond_df, cond_df.shape[1]

# ---------------------------------------------------------------------------
# Sequences & scaling
# ---------------------------------------------------------------------------
def build_sequences(y: np.ndarray, c: np.ndarray, seq_len: int = 60) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build rolling sequences for supervised learning:
    - X: (N', seq_len, 1)
    - C: (N', seq_len, F)
    - t: (N', 1)
    """
    X, Cseq, t = [], [], []
    N = len(y)
    for i in range(seq_len, N):
        X.append(y[i - seq_len:i].reshape(-1, 1))
        Cseq.append(c[i - seq_len:i, :])
        t.append(y[i])
    X = np.array(X, dtype=np.float32)
    Cseq = np.array(Cseq, dtype=np.float32)
    t = np.array(t, dtype=np.float32).reshape(-1, 1)
    return X, Cseq, t

def scale_fit_transform(train_arr: np.ndarray, val_arr: np.ndarray, test_arr: np.ndarray, is_2d: bool = False):
    """
    Fit a StandardScaler on train_arr and transform val_arr and test_arr.
    - If is_2d=False, expects shape (N,) and returns flattened output arrays.
    - If is_2d=True, expects shape (N, F) and returns 2D arrays.
    """
    scaler = StandardScaler()
    if not is_2d:
        train_arr = train_arr.reshape(-1, 1)
        val_arr = val_arr.reshape(-1, 1)
        test_arr = test_arr.reshape(-1, 1)
    tr = scaler.fit_transform(train_arr)
    va = scaler.transform(val_arr)
    te = scaler.transform(test_arr)
    if not is_2d:
        return scaler, tr.ravel(), va.ravel(), te.ravel()
    else:
        return scaler, tr.astype(np.float32), va.astype(np.float32), te.astype(np.float32)

def get_raw_planetary_positions(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Public wrapper to get raw planetary longitudes (0..360) indexed by dates.
    """
    longs = _planet_positions(dates)
    df_astro = pd.DataFrame(longs, index=dates)
    df_astro = df_astro.fillna(method='ffill').fillna(method='bfill')
    return df_astro

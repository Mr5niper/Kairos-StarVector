# Kairos Vector — README

A complete forecasting system with:

- Walk-forward, leakage-safe evaluation
- Models:
  - ARIMA (baseline)
  - Conditional LSTM (DPA-optimized early stopping)
  - Conditional WGAN-GP (stable cGAN; DPA-optimized early stopping)
  - Residual Fusion (LightGBM + residual cWGAN-GP)
  - Meta-labeling filter (LightGBM) to accept/reject GAN signals
- Real conditional features:
  - Astro (Skyfield): CSW/Bradley-like daily scores; Gann proximity proxy
  - Event/Sentiment (RSS → Sentence-BERT + SinglePass + FinBERT)
- Statistical significance: Diebold–Mariano tests (GAN vs LSTM, GAN vs ARIMA)
- Confidence intervals over windows
- Backtest (long/short with transaction costs)
- GUI (Streamlit) and CLI modes
- Optional: Optuna hyperparameter tuning across multiple windows

MIT Licensed.

---

## 1) Install

```bash
# Create and activate a virtualenv (optional but recommended)
python -m venv venv
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate

# Install dependencies (first run downloads models/ephemeris on demand)
pip install -r requirements.txt
```

---

## 2) Data Preparation (News / Features)

This framework can run in two modes:

- `features.mode: "real"` (default): Builds real conditional features from news + astro
- `features.mode: "dummy"`: Uses synthetic features (for quick smoke tests)

### Option A: Fetch News via RSS (recommended)

```bash
# Fetch headlines from Reuters, CNBC, MarketWatch, Yahoo Finance RSS (broad coverage).
python scripts/fetch_news_rss.py --out features/news_headlines.csv --start 2015-01-01 --end 2025-01-01
```

Expected CSV schema:
```
Date,Title
2019-08-05,Markets tumble as trade tensions rise
2019-08-05,Fed officials weigh rate cuts
...
```

> Note: RSS coverage is limited by source feed history. For best results,
> supply your own long-horizon news CSV with a daily `Date` and `Title`.

### Option B: Use Your Own News File

- Place a CSV at `features/news_headlines.csv` with columns `Date,Title`.
- Ensure `Date` is ISO format (YYYY-MM-DD), consistent with the market’s localtz.

---

## 3) Build Conditional Features (Astro + Event/Sentiment)

```bash
python scripts/build_features.py
```

This will compute and cache daily features to `artifacts/cond_features_cache.csv`:
- Price/Volatility context (ATR proxy, MA ratio, returns lag)
- Astro (Skyfield): `astro_csw`, `astro_bradley`, `astro_gann_prox`
- Event/Sentiment (SBERT + FinBERT): `event_heat`, `event_heat_neg`, `event_sentiment`

> First run will download:
> - Skyfield DE421 ephemeris (~20MB)
> - SBERT and FinBERT models (~hundreds of MB)
> Ensure you have internet access on first run.

---

## 4) Run the GUI (Streamlit)

```bash
streamlit run gui/streamlit_app.py
```

What you get:
- Left sidebar: data range, feature mode, news CSV path, astro aspect controls, model options (Residual, Meta), and transaction costs
- Buttons:
  - “Build Features Cache (Real Mode)” — compute and cache conditional features
  - “Run Benchmark” — train and evaluate across rolling windows
- Tabs:
  - Dashboard (KPIs, model comparison table, equity curves)
  - Predictions (interactive chart for first test window)
  - Feature Explorer (visualize cached features)
  - Windows (per-window metrics table)
  - Logs (progress/info)

---

## 5) Run from CLI (Optional)

If you prefer command-line:

```bash
python scripts/run_benchmark.py
```

Aggregated metrics saved to:
```
artifacts/summary.json
```

Per-window predictions saved to:
```
artifacts/preds_window_<n>.csv
```

---

## 6) Configuration (configs/default.yaml)

Key sections:

```yaml
data:
  ticker: "^GSPC"          # Yahoo symbol
  start: "2015-01-01"
  end: null
  seq_len: 60
  target: "log_return"

features:
  mode: "real"             # "real" uses RSS+SBERT+FinBERT+Skyfield; "dummy" uses synthetic
  news_csv_path: "features/news_headlines.csv"
  news_date_col: "Date"
  news_text_col: "Title"
  singlepass_threshold: 0.72
  aspect_orb_deg: 3.0
  aspects_deg: [0, 60, 90, 120, 180]
  cache_path: "artifacts/cond_features_cache.csv"

splits:
  train_len: 756           # ~3y
  val_len: 126             # ~6m
  test_len: 126            # ~6m
  step: 126

model:
  lstm: { val_metric: "DPA", ... }
  gan:  { val_metric: "DPA", ... }
  residual:
    enabled: true

meta_labeling:
  enabled: true

ablation:
  residual_compare: true
  meta_compare: true
```

---

## 7) How It Works

- **Real conditional features**:
  - **Astro**: Compute daily planetary longitudes (Skyfield), aggregate aspect alignments (CSW-like), cos(separation) sums (Bradley-like), proximity to round-number “Gann” price grid
  - **Event/Sentiment**: Aggregate daily news titles via SBERT embeddings into topic clusters (SinglePass), evaluate sentiment using FinBERT, compute daily `event_heat`, `event_heat_neg`, `event_sentiment`
- **Models**:
  - **ARIMA** on raw returns
  - **LSTM (Cond)** optimized for **DPA** (direction sign accuracy), early stopping on val DPA
  - **cWGAN-GP (Cond)** stabilized on returns; early stopping on val DPA
  - **Residual Fusion**: LightGBM on last-step features + cWGAN-GP on residuals
  - **Meta-Labeling**: LightGBM classifier filters GAN trades based on features + predicted return
- **Evaluation**:
  - RMSE, MAE, MAPE, DPA
  - Diebold–Mariano tests (GAN vs LSTM, GAN vs ARIMA)
  - 95% CI on aggregated metrics
- **Backtest**:
  - Long if pred > 0, otherwise short
  - Transaction costs applied on signal flips (bps)

---

## 8) Quick Start (TL;DR)

```bash
# 1) Install
pip install -r requirements.txt

# 2) Fetch news (RSS) – optional if you already have features/news_headlines.csv
python scripts/fetch_news_rss.py --out features/news_headlines.csv --start 2015-01-01 --end 2025-01-01

# 3) Build features (astro + event)
python scripts/build_features.py

# 4) GUI
streamlit run gui/streamlit_app.py

# OR: CLI
python scripts/run_benchmark.py
```

---

## 9) Tips for Speed / First Run

- First run downloads models (SBERT, FinBERT) and ephemeris (Skyfield).
- To test pipeline quickly:
  - Set `features.mode: "dummy"`
  - Narrow `data.start` (e.g., last 2–3 years)
  - Disable `residual.enabled` and `meta_labeling.enabled`
  - Reduce `model.*.epochs`
- GPU recommended (PyTorch) for faster deep model training.

---

## 10) Tuning (Optional; Multi-Window)

```bash
# Enable in configs/default.yaml:
tuning:
  enable: true
  n_trials: 30
  windows: 3
  target_metric: "DPA"

# Run Optuna
python scripts/run_optuna.py
```

This tunes LSTM/GAN hyperparams by maximizing average DPA across multiple walk-forward windows.

---

## 11) Troubleshooting

- **“No valid news parsed”**: Use `scripts/fetch_news_rss.py` or supply your own `features/news_headlines.csv`.
- **Skyfield ephemeris not found**: Program downloads ephemeris on first run. Ensure internet access.
- **HuggingFace model download fails**: Check firewall/proxy settings; you can pre-download models.
- **Memory issues (GPU)**: Reduce `batch_size`, `epochs`, or switch `device` to CPU.

---

## 12) Folder Structure

```
.
├─ configs/
│  └─ default.yaml
├─ features/
│  └─ news_headlines.csv         # daily headlines (Date, Title)
├─ artifacts/
│  ├─ cond_features_cache.csv    # daily conditional features (built)
│  ├─ preds_window_*.csv         # per-window predictions
│  └─ summary.json               # aggregated metrics
├─ stock_forecast/
│  ├─ dataset.py                 # astro & event pipelines; scaling; sequences
│  ├─ models/ (arima, lstm, cgan)
│  ├─ train_lstm.py              # DPA early stopping
│  ├─ train_gan.py               # DPA early stopping + best checkpoint
│  ├─ meta_labeling.py           # meta filter
│  ├─ backtest.py, eval.py, metrics.py, splits.py, utils.py
├─ scripts/
│  ├─ build_features.py          # builds & caches conditional features
│  ├─ fetch_news_rss.py          # RSS news fetcher (Date, Title)
│  ├─ run_benchmark.py           # CLI evaluation
│  └─ run_optuna.py              # hyperparameter tuning
└─ gui/
   └─ streamlit_app.py           # pro GUI
```

---

## 13) License

MIT

---

## 14) Credits

- FinBERT: ProsusAI/finbert
- Sentence-BERT: sentence-transformers/all-MiniLM-L6-v2
- Skyfield: NASA/JPL ephemerides
- Inspired by CCIR best practices (conditional models, event-aware signals) and SOTA WGAN‑GP stability for time series.
```
```markdown
# QUICKSTART.md

## 1) Install

```bash
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

## 2) Fetch News (RSS)

```bash
python scripts/fetch_news_rss.py --out features/news_headlines.csv --start 2015-01-01 --end 2025-01-01
```

## 3) Build Features (Astro + Event/Sentiment)

```bash
python scripts/build_features.py
```

This writes: `artifacts/cond_features_cache.csv`.

## 4) Run GUI

```bash
streamlit run gui/streamlit_app.py
```

- Configure ticker/date range/features in the sidebar.
- Click “Build Features Cache (Real Mode)” if not already cached.
- Click “Run Benchmark”.

## 5) Results

- Aggregated metrics: `artifacts/summary.json`
- Per-window predictions: `artifacts/preds_window_*.csv`
- GUI tabs include KPIs, model comparison, equity curves, feature explorer, window tables, logs.

## 6) Optional CLI

```bash
python scripts/run_benchmark.py
```

## 7) Optional Tuning (Multi-Window Optuna)

```bash
# In configs/default.yaml: tuning.enable: true
python scripts/run_optuna.py
```

## 8) Notes

- First run downloads FinBERT/SBERT and Skyfield ephemeris (internet required).
- To test quickly: set `features.mode: "dummy"` in config, reduce date range and epochs.
- Meta-labeling and Residual Fusion can be toggled in sidebar/ YAML.

MIT License.
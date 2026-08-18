"""
kairos.market
=============
Price data loading.

The bug that broke the original build
-------------------------------------
`yfinance` returns a MultiIndex column frame — ("Close", "^GSPC") rather
than plain "Close" — even for a single ticker. So `df['Close']` handed
back a one-column DataFrame instead of a Series, and every downstream
consumer that expected a 1-D array received shape (N, 1). The old
`dataset.py` was full of defensive `.reshape(-1)` calls and a comment
reading "fixes ValueError: shape (N,1)", which is the symptom being
patched over rather than the cause being fixed.

Fixed once, here, at the boundary: columns are flattened, the index is
made timezone-naive, and column names are normalised before anything else
sees the data. Nothing downstream needs to defend itself.

Also added: a disk cache so the app works offline after the first fetch,
and a synthetic series so the GUI still renders when there is no network
at all instead of dying on an empty frame.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from .paths import cache_dir

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns, drop the timezone, normalise names."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=OHLCV)

    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        # yfinance orders levels as (Price, Ticker); level 0 holds the
        # OHLCV name. Fall back to the level that actually contains them.
        lvl0 = {str(c) for c in out.columns.get_level_values(0)}
        level = 0 if lvl0 & set(OHLCV) else 1
        out.columns = [str(c[level]) for c in out.columns]
    else:
        out.columns = [str(c) for c in out.columns]

    rename = {}
    for c in out.columns:
        key = c.strip().lower().replace(" ", "").replace("_", "")
        for want in OHLCV:
            if key == want.lower():
                rename[c] = want
        if key in ("adjclose", "adjustedclose") and "Close" not in out.columns:
            rename[c] = "Close"
    out = out.rename(columns=rename)
    out = out.loc[:, ~out.columns.duplicated()]

    out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_convert("UTC").tz_localize(None)
    out.index = out.index.normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()

    for col in OHLCV:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out[OHLCV].dropna(subset=["Close"])
    out.index.name = "Date"
    return out


def _cache_path(ticker: str, interval: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in ticker)
    return os.path.join(cache_dir(), f"{safe}_{interval}.csv")


def save_cache(df: pd.DataFrame, ticker: str, interval: str = "1d") -> None:
    try:
        path = _cache_path(ticker, interval)
        out = df.copy()
        out.insert(0, "Date", out.index)
        out.to_csv(path, index=False)
    except Exception:
        pass


def load_cache(ticker: str, interval: str = "1d") -> Optional[pd.DataFrame]:
    path = _cache_path(ticker, interval)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
        return _normalise(df)
    except Exception:
        return None


def fetch_ohlc(
    ticker: str,
    start: str = "2015-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
    use_cache: bool = True,
    allow_cache_fallback: bool = True,
) -> pd.DataFrame:
    """
    Download OHLCV for one ticker, already flattened and cleaned.

    Tries `Ticker.history` first, which returns single-level columns, then
    falls back to `download`. On failure, returns cached data if any is
    available rather than raising, because a stale chart beats a stack
    trace in a GUI.
    """
    err: Optional[Exception] = None
    df = pd.DataFrame()

    try:
        import yfinance as yf
    except Exception as exc:                       # pragma: no cover
        raise RuntimeError(
            "yfinance is not installed. Run: pip install -r requirements.txt"
        ) from exc

    try:
        tk = yf.Ticker(ticker)
        df = tk.history(start=start, end=end, interval=interval, auto_adjust=True)
        df = _normalise(df)
    except Exception as exc:
        err = exc
        df = pd.DataFrame()

    if df.empty:
        try:
            raw = yf.download(
                ticker, start=start, end=end, interval=interval,
                auto_adjust=True, progress=False, threads=False,
            )
            df = _normalise(raw)
        except Exception as exc:
            err = err or exc
            df = pd.DataFrame()

    if not df.empty:
        if use_cache:
            save_cache(df, ticker, interval)
        return df

    if allow_cache_fallback:
        cached = load_cache(ticker, interval)
        if cached is not None and not cached.empty:
            sliced = cached.loc[
                (cached.index >= pd.Timestamp(start)) &
                (cached.index <= (pd.Timestamp(end) if end else cached.index.max()))
            ]
            if not sliced.empty:
                return sliced

    detail = f" Last error: {err}" if err else ""
    raise RuntimeError(
        f"No data returned for '{ticker}' between {start} and {end or 'today'}. "
        f"Check the symbol spelling, the date range, and your connection.{detail}"
    )


def demo_ohlc(
    start: str = "2015-01-01",
    end: Optional[str] = None,
    seed: int = 7,
    start_price: float = 100.0,
) -> pd.DataFrame:
    """
    Synthetic random-walk OHLCV, so every chart and statistic in the app
    can be exercised without a network connection.

    Doubles as the control case: this series has no relationship to
    planetary geometry by construction, so whatever the analysis tabs
    report here is the noise floor for your settings.
    """
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    idx = pd.bdate_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize())
    if len(idx) == 0:
        idx = pd.bdate_range(pd.Timestamp(start).normalize(), periods=250)

    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0003, 0.011, len(idx))
    close = float(start_price) * np.exp(np.cumsum(ret))
    noise = np.abs(rng.normal(0, 0.006, len(idx)))

    df = pd.DataFrame({
        "Open": close * (1 - noise * 0.5),
        "High": close * (1 + noise),
        "Low": close * (1 - noise),
        "Close": close,
        "Volume": rng.integers(1_000_000, 9_000_000, len(idx)).astype(float),
    }, index=idx)
    df.index.name = "Date"
    return _normalise(df)


def log_returns(close: pd.Series) -> pd.Series:
    s = pd.to_numeric(close, errors="coerce").dropna()
    return np.log(s).diff().dropna()


def describe(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {}
    c = df["Close"]
    return {
        "bars": int(len(df)),
        "first": str(df.index.min().date()),
        "last": str(df.index.max().date()),
        "first_close": round(float(c.iloc[0]), 4),
        "last_close": round(float(c.iloc[-1]), 4),
        "min_close": round(float(c.min()), 4),
        "max_close": round(float(c.max()), 4),
        "total_return_pct": round(float((c.iloc[-1] / c.iloc[0] - 1) * 100), 2),
    }

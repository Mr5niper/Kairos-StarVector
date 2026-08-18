"""
kairos.waves
============
Turns discrete alignment events into a continuous wave, finds the cycles
actually present in price, and measures whether the two are related.

The premise this module implements
----------------------------------
The idea driving this project is that an alignment does not act on price
on the day it happens. It nudges some decisions, those decisions change
other people's information and positions, and the effect spreads forward
in time, decaying as it goes. That is a convolution: an impulse train
(the alignment dates, weighted by strength) convolved with a decay
kernel. Sum the overlapping tails and you get a wave.

That is a real, testable model, and it is what `composite_pressure`
builds.

On the statistics
-----------------
Both price and the composite wave are heavily autocorrelated. A plain
Pearson correlation between two smooth series will look impressive
almost regardless of whether any relationship exists, and its textbook
p-value will be wrong by orders of magnitude. Two smooth random walks
routinely correlate at 0.8.

So every test here uses a surrogate method that preserves
autocorrelation: phase randomisation for the continuous comparisons,
block bootstrap and date permutation for the event study. If a result
survives those, it is worth a second look. If it does not, the naive
correlation was an artefact of smoothness, not a signal. The point of
the module is to tell you which one you are looking at.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252.0

# Fewest paired observations a correlation is computed from. Below this the
# number is dominated by whichever handful of points survived the shift.
MIN_OVERLAP = 32


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def zscore(x, ddof: int = 0) -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    sd = np.nanstd(a, ddof=ddof)
    if not np.isfinite(sd) or sd < 1e-15:
        return np.zeros_like(a)
    return (a - np.nanmean(a)) / sd


def detrend(x, mode: str = "linear") -> np.ndarray:
    """Remove trend so the spectrum is not dominated by the trend itself."""
    a = np.asarray(x, dtype=float).ravel()
    if mode == "none":
        return a - np.nanmean(a)
    if mode == "diff":
        return np.diff(a, prepend=a[0])
    t = np.arange(len(a), dtype=float)
    good = np.isfinite(a)
    if good.sum() < 3:
        return a - np.nanmean(a)
    coef = np.polyfit(t[good], a[good], 1)
    return a - np.polyval(coef, t)


def to_calendar_daily(s: pd.Series) -> pd.Series:
    """
    Reindex a trading-day series onto every calendar day, forward filling
    weekends and holidays.

    Necessary because planetary periods are in calendar days. A cycle
    measured in trading-day units cannot be compared with a 236.99-day
    synodic period without this step, and mixing the two is an easy way
    to "discover" cycles that are pure calendar artefacts.
    """
    s = s.dropna()
    if s.empty:
        return s
    idx = pd.date_range(s.index.min().normalize(), s.index.max().normalize(), freq="D")
    return s.reindex(idx).ffill()


# --------------------------------------------------------------------------
# The trickle-down kernel
# --------------------------------------------------------------------------
def trickle_kernel(
    horizon_days: int,
    tau_days: float,
    period_days: Optional[float] = None,
    phase_deg: float = 0.0,
    lead_days: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Impulse response of a single alignment event.

    horizon_days : how far forward the influence is allowed to reach
    tau_days     : exponential decay constant; the influence falls to 37%
                   of its initial value after this many days
    period_days  : if set, the decaying tail also oscillates with this
                   period, which is what produces a wave rather than a
                   simple decaying bump
    lead_days    : anticipation before the event, rising to the peak

    Returns (lags, weights) where lags may be negative when lead_days > 0.
    """
    tau = max(float(tau_days), 0.5)
    lead = max(int(lead_days), 0)
    lags = np.arange(-lead, int(max(horizon_days, 1)) + 1, dtype=float)

    env = np.where(lags >= 0,
                   np.exp(-np.clip(lags, 0, None) / tau),
                   np.exp(-np.abs(np.clip(lags, None, 0)) / max(tau * 0.4, 0.5)))

    if period_days and float(period_days) > 1.0:
        env = env * np.cos(2.0 * np.pi * lags / float(period_days) + np.deg2rad(phase_deg))

    return lags, env


def composite_pressure(
    index: pd.DatetimeIndex,
    events: pd.DataFrame,
    tau_days: float = 21.0,
    horizon_days: int = 120,
    period_days: Optional[float] = None,
    phase_deg: float = 0.0,
    lead_days: int = 0,
    date_col: str = "date",
    weight_col: str = "weight",
    normalise: bool = True,
) -> pd.Series:
    """
    Sum the decaying tails of every alignment event onto a daily index.

    This is the core of the model: overlapping tails from many events at
    different strengths interfere, and the sum is the "pressure" wave. It
    is deterministic and can be extended past the last price bar, because
    future alignment dates are known exactly.
    """
    index = pd.DatetimeIndex(index)
    out = np.zeros(len(index), dtype=float)
    if events is None or len(events) == 0 or len(index) == 0:
        return pd.Series(out, index=index, name="astro_pressure")

    lags, kern = trickle_kernel(horizon_days, tau_days, period_days, phase_deg, lead_days)

    # Map event dates onto positions in the index. searchsorted keeps this
    # O(n log n) instead of a Python loop over every event-day pair.
    ev = events.dropna(subset=[date_col]).copy()
    ev_dates = pd.DatetimeIndex(pd.to_datetime(ev[date_col])).normalize()
    idx_norm = index.normalize()
    pos = np.searchsorted(idx_norm.values, ev_dates.values, side="left")

    weights = (ev[weight_col].to_numpy(dtype=float)
               if weight_col in ev.columns else np.ones(len(ev), dtype=float))

    n = len(index)
    for p, w in zip(pos, weights):
        if not np.isfinite(w) or w == 0.0:
            continue
        targets = p + lags.astype(int)
        keep = (targets >= 0) & (targets < n)
        if not keep.any():
            continue
        np.add.at(out, targets[keep], w * kern[keep])

    if normalise:
        sd = np.std(out)
        if sd > 1e-12:
            out = (out - out.mean()) / sd

    return pd.Series(out, index=index, name="astro_pressure")


# --------------------------------------------------------------------------
# Cycle detection
# --------------------------------------------------------------------------
def dominant_cycles(
    series: pd.Series,
    min_period_days: float = 20.0,
    max_period_days: float = 3000.0,
    n_top: int = 12,
    detrend_mode: str = "linear",
) -> pd.DataFrame:
    """
    Periodogram peaks of a calendar-daily series.

    Returns period in calendar days, spectral power, power as a share of
    total in-band power, and the number of whole cycles the sample
    contains. That last column matters: a "cycle" seen 1.2 times in the
    sample is not evidence of anything, and this is exactly how spurious
    long cycles get reported.
    """
    s = to_calendar_daily(series.dropna())
    n = len(s)
    if n < 64:
        return pd.DataFrame(columns=["period_days", "period_years", "power",
                                     "power_share", "cycles_in_sample"])

    x = detrend(s.to_numpy(dtype=float), detrend_mode)
    x = x * np.hanning(n)                       # reduce spectral leakage
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0)           # cycles per day

    with np.errstate(divide="ignore"):
        periods = np.where(freqs > 0, 1.0 / freqs, np.inf)

    band = (periods >= min_period_days) & (periods <= min(max_period_days, n / 2.0))
    if not band.any():
        return pd.DataFrame(columns=["period_days", "period_years", "power",
                                     "power_share", "cycles_in_sample"])

    p_band, s_band = periods[band], spec[band]
    total = s_band.sum()

    # Local maxima only, so one broad peak is not reported as five.
    if len(s_band) >= 3:
        is_peak = np.zeros(len(s_band), dtype=bool)
        is_peak[1:-1] = (s_band[1:-1] > s_band[:-2]) & (s_band[1:-1] > s_band[2:])
    else:
        is_peak = np.ones(len(s_band), dtype=bool)

    order = np.argsort(s_band[is_peak])[::-1][:int(n_top)]
    pk_p, pk_s = p_band[is_peak][order], s_band[is_peak][order]

    return pd.DataFrame({
        "period_days": np.round(pk_p, 2),
        "period_years": np.round(pk_p / 365.2422, 3),
        "power": pk_s,
        "power_share": np.round(pk_s / total, 4) if total > 0 else 0.0,
        "cycles_in_sample": np.round(n / pk_p, 2),
    }).reset_index(drop=True)


def spectrum(
    series: pd.Series,
    min_period_days: float = 20.0,
    max_period_days: float = 3000.0,
    detrend_mode: str = "linear",
) -> pd.DataFrame:
    """Full in-band periodogram, for plotting."""
    s = to_calendar_daily(series.dropna())
    n = len(s)
    if n < 64:
        return pd.DataFrame(columns=["period_days", "power"])
    x = detrend(s.to_numpy(dtype=float), detrend_mode) * np.hanning(n)
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0)
    with np.errstate(divide="ignore"):
        periods = np.where(freqs > 0, 1.0 / freqs, np.inf)
    band = (periods >= min_period_days) & (periods <= min(max_period_days, n / 2.0))
    df = pd.DataFrame({"period_days": periods[band], "power": spec[band]})
    return df.sort_values("period_days").reset_index(drop=True)


def match_cycles_to_synodics(
    cycles: pd.DataFrame,
    synodics: pd.DataFrame,
    tolerance_pct: float = 5.0,
    include_harmonics: Sequence[int] = (1, 2, 3, 4),
) -> pd.DataFrame:
    """
    For each detected price cycle, the closest synodic period or one of
    its harmonics.

    Harmonics are included because a 237-day Venus-Jupiter cycle would
    also express as roughly 118 or 79 days if the market responded to
    half or a third of the cycle. That flexibility cuts both ways: the
    more harmonics allowed, the easier it is to find a match by chance,
    which is why the miss distance is reported rather than hidden.
    """
    if cycles.empty or synodics.empty:
        return pd.DataFrame(columns=["period_days", "best_match", "harmonic",
                                     "synodic_days", "match_period", "miss_pct", "within_tolerance"])
    rows = []
    for _, c in cycles.iterrows():
        p = float(c["period_days"])
        best = None
        for _, s in synodics.iterrows():
            for h in include_harmonics:
                cand = float(s["synodic_days"]) / float(h)
                miss = abs(cand - p) / p * 100.0
                if best is None or miss < best["miss_pct"]:
                    best = {
                        "period_days": round(p, 2),
                        "best_match": s["pair"],
                        "harmonic": int(h),
                        "synodic_days": round(float(s["synodic_days"]), 2),
                        "match_period": round(cand, 2),
                        "miss_pct": round(miss, 2),
                    }
        if best:
            best["within_tolerance"] = bool(best["miss_pct"] <= tolerance_pct)
            best["power_share"] = float(c.get("power_share", np.nan))
            best["cycles_in_sample"] = float(c.get("cycles_in_sample", np.nan))
            rows.append(best)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Significance testing
# --------------------------------------------------------------------------
def _phase_randomise(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Surrogate series with the same power spectrum and autocorrelation as
    x, but randomised phases and therefore no relationship to anything
    else. The standard null model for correlating two smooth series.
    """
    n = len(x)
    fx = np.fft.rfft(x)
    mag = np.abs(fx)
    phases = rng.uniform(0, 2 * np.pi, len(fx))
    phases[0] = 0.0
    if n % 2 == 0 and len(phases) > 1:
        phases[-1] = 0.0
    return np.fft.irfft(mag * np.exp(1j * phases), n=n)


def surrogate_correlation(
    a: pd.Series,
    b: pd.Series,
    n_surrogates: int = 500,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Correlation between two series with a p-value that accounts for
    autocorrelation.

    Reports both the naive p-value and the surrogate p-value so the gap
    between them is visible. That gap is usually large, and it is the
    whole reason this function exists.
    """
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(joined) < 32:
        return {"n": len(joined), "pearson": np.nan, "spearman": np.nan,
                "p_naive": np.nan, "p_surrogate": np.nan}

    x = joined["a"].to_numpy(dtype=float)
    y = joined["b"].to_numpy(dtype=float)
    n = len(x)

    r = float(np.corrcoef(zscore(x), zscore(y))[0, 1])

    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    rho = float(np.corrcoef(rx, ry)[0, 1])

    # Naive two-sided t test, which assumes independent observations.
    try:
        from scipy import stats
        t = r * np.sqrt(max(n - 2, 1) / max(1 - r * r, 1e-12))
        p_naive = float(2 * (1 - stats.t.cdf(abs(t), df=max(n - 2, 1))))
    except Exception:
        p_naive = np.nan

    rng = np.random.default_rng(seed)
    xd = detrend(x, "linear")
    yz = zscore(detrend(y, "linear"))
    hits = 0
    for _ in range(int(n_surrogates)):
        surr = zscore(_phase_randomise(xd, rng))
        rs = float(np.corrcoef(surr, yz)[0, 1])
        if abs(rs) >= abs(r):
            hits += 1
    p_surr = (hits + 1) / (int(n_surrogates) + 1)

    return {"n": n, "pearson": r, "spearman": rho,
            "p_naive": p_naive, "p_surrogate": float(p_surr)}


def lead_lag_correlation(
    a: pd.Series,
    b: pd.Series,
    max_lag_days: int = 120,
) -> pd.DataFrame:
    """
    Correlation of a against b shifted by a range of lags.

    A positive best lag means the alignment wave leads price, which is
    the shape the trickle-down idea predicts. Note that scanning many
    lags and keeping the best inflates the apparent correlation, so the
    best lag should be confirmed with `surrogate_correlation` at that lag
    rather than trusted on its own.
    """
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(joined) < 64:
        return pd.DataFrame(columns=["lag_days", "correlation"])
    x = zscore(detrend(joined["a"].to_numpy(dtype=float), "linear"))
    y = zscore(detrend(joined["b"].to_numpy(dtype=float), "linear"))
    n = len(x)

    # Clamp the scan to what the sample can actually support. Asking for a
    # 400-day lag on 224 days of data is not a smaller answer, it is no
    # answer, and the arithmetic below fails silently rather than loudly if
    # allowed to go out of range: at lag > n, `len(x) - lag` is negative and
    # `x[:-76]` returns a slice from the FRONT of the array instead of an
    # empty one, while `y[lag:]` correctly returns nothing. The result is a
    # 148-element array paired with a 0-element one, and numpy raises deep
    # inside np.cov where the cause is unrecognisable.
    max_lag = int(min(int(max_lag_days), n - MIN_OVERLAP))
    if max_lag < 1:
        return pd.DataFrame(columns=["lag_days", "correlation"])

    lags, corrs = [], []
    for lag in range(-max_lag, max_lag + 1):
        # Slice bounds are clipped into range before use, so the pair can
        # never be built with mismatched lengths in the first place.
        if lag < 0:
            xa, ya = x[min(-lag, n):], y[:max(n + lag, 0)]
        elif lag > 0:
            xa, ya = x[:max(n - lag, 0)], y[min(lag, n):]
        else:
            xa, ya = x, y

        m = min(len(xa), len(ya))
        if m < MIN_OVERLAP:
            continue
        xa, ya = xa[:m], ya[:m]

        # A constant slice makes the correlation undefined; numpy returns
        # NaN and warns rather than raising.
        if np.std(xa) < 1e-12 or np.std(ya) < 1e-12:
            continue

        lags.append(lag)
        corrs.append(float(np.corrcoef(xa, ya)[0, 1]))

    return pd.DataFrame({"lag_days": lags, "correlation": corrs})


def event_study(
    close: pd.Series,
    event_dates: Iterable,
    horizons: Sequence[int] = (1, 3, 5, 10, 21),
    n_permutations: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Average forward return after alignment dates, against a permutation
    null.

    The null draws the same number of random dates from the same price
    series and recomputes the mean, repeatedly. The p-value is the share
    of random draws that beat the observed mean in absolute terms. This
    controls for sample size, for the drift in the underlying series, and
    for the volatility of the period, none of which a raw average does.

    A note on reading the output: with five horizons tested, one showing
    p < 0.05 is roughly what pure chance produces. Look for consistency
    across horizons, not for the single best cell.
    """
    close = close.dropna()
    if len(close) < 60:
        return pd.DataFrame(columns=["horizon_days", "n_events", "mean_return",
                                     "baseline_return", "excess", "hit_rate",
                                     "p_permutation"])

    logp = np.log(close.to_numpy(dtype=float))
    idx = pd.DatetimeIndex(close.index).normalize()

    ev = pd.DatetimeIndex(pd.to_datetime(list(event_dates))).normalize()
    pos = np.searchsorted(idx.values, ev.values, side="left")
    pos = np.unique(pos[(pos >= 0) & (pos < len(logp))])

    rng = np.random.default_rng(seed)
    rows = []
    for h in horizons:
        h = int(h)
        valid = pos[pos + h < len(logp)]
        if len(valid) < 5:
            continue
        fwd = logp[valid + h] - logp[valid]
        obs = float(np.mean(fwd))

        all_start = np.arange(0, len(logp) - h)
        base_all = logp[all_start + h] - logp[all_start]
        baseline = float(np.mean(base_all))

        k = len(valid)
        perm_means = np.empty(int(n_permutations), dtype=float)
        for i in range(int(n_permutations)):
            pick = rng.choice(all_start, size=k, replace=False)
            perm_means[i] = np.mean(logp[pick + h] - logp[pick])

        centred_obs = obs - baseline
        centred_perm = perm_means - baseline
        p = float((np.sum(np.abs(centred_perm) >= abs(centred_obs)) + 1) / (int(n_permutations) + 1))

        rows.append({
            "horizon_days": h,
            "n_events": int(k),
            "mean_return": round(obs, 6),
            "baseline_return": round(baseline, 6),
            "excess": round(centred_obs, 6),
            "hit_rate": round(float(np.mean(fwd > 0)), 4),
            "p_permutation": round(p, 4),
        })
    return pd.DataFrame(rows)


def block_bootstrap_mean(
    x: np.ndarray,
    block: int = 21,
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Mean with a confidence interval that survives autocorrelation, by
    resampling contiguous blocks instead of individual observations.
    Returns (mean, lo95, hi95).
    """
    a = np.asarray(x, dtype=float).ravel()
    a = a[np.isfinite(a)]
    n = len(a)
    if n < block * 2:
        return (float(np.mean(a)) if n else np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(int(n_boot), dtype=float)
    starts_pool = np.arange(0, n - block + 1)
    for i in range(int(n_boot)):
        starts = rng.choice(starts_pool, size=n_blocks, replace=True)
        sample = np.concatenate([a[s:s + block] for s in starts])[:n]
        means[i] = sample.mean()
    return float(a.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarise_relationship(
    close: pd.Series,
    pressure: pd.Series,
    events: pd.DataFrame,
    max_lag_days: int = 120,
    n_surrogates: int = 300,
    horizons: Sequence[int] = (1, 3, 5, 10, 21),
    n_permutations: int = 500,
) -> Dict[str, object]:
    """
    One call that runs the full honest battery and returns everything the
    GUI needs to report.
    """
    close = close.dropna()
    daily_close = to_calendar_daily(close)
    press = pressure.reindex(daily_close.index).ffill()

    level = surrogate_correlation(press, np.log(daily_close), n_surrogates=n_surrogates)

    ret = np.log(daily_close).diff().dropna()
    fwd21 = np.log(daily_close).shift(-21) - np.log(daily_close)
    ret_corr = surrogate_correlation(press.reindex(ret.index), ret, n_surrogates=n_surrogates)
    fwd_corr = surrogate_correlation(press, fwd21.dropna(), n_surrogates=n_surrogates)

    ll = lead_lag_correlation(press, np.log(daily_close), max_lag_days=max_lag_days)
    best_lag, best_corr = (np.nan, np.nan)
    if not ll.empty:
        j = int(ll["correlation"].abs().idxmax())
        best_lag = int(ll.loc[j, "lag_days"])
        best_corr = float(ll.loc[j, "correlation"])

    es = pd.DataFrame()
    if events is not None and len(events) > 0 and "date" in events.columns:
        es = event_study(close, events["date"], horizons=horizons,
                         n_permutations=n_permutations)

    return {
        "level_correlation": level,
        "return_correlation": ret_corr,
        "forward_21d_correlation": fwd_corr,
        "lead_lag": ll,
        "best_lag_days": best_lag,
        "best_lag_correlation": best_corr,
        "event_study": es,
    }

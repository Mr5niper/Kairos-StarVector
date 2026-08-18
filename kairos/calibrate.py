"""
kairos.calibrate
================
Fit the wave parameters on past data, check them on data the fit never saw,
then project a signal forward.

The workflow this implements
----------------------------
1. Split history into TRAIN and TEST at a date you choose.
2. Search wave parameters — bodies, aspects, orb, decay, oscillation,
   signal rule — scoring each candidate on TRAIN only.
3. Re-score the winners on TEST, which the search never touched.
4. Apply the chosen parameters forward, past the last price bar, and emit
   dated long/short/flat positions.

Step 3 is the one that makes the rest worth anything, and it is not
optional here. A search over a few hundred parameter combinations will
always find something that fits TRAIN well; that is what searching does.
The question is whether it still works on data it has not seen. So every
result carries three numbers: the train score, the test score, and the
gap between them.

Read the gap, not the train score. A candidate scoring 0.58 on train and
0.51 on test found a pattern in the training window and nothing else. One
scoring 0.54 on both is far more interesting, even though it looks worse.

One legitimate advantage worth knowing about
--------------------------------------------
`signal_offset` may be POSITIVE, meaning the signal uses the wave value
from N days in the future. That is not lookahead cheating and it is the
one real edge this whole approach has: planetary positions are computable
decades ahead, so the wave's future values are genuinely known today in a
way no price-derived indicator's are.

Future PRICE is never touched anywhere in this module. Only the wave is
read ahead, and only ever the wave.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import astro as A
from . import waves as W

TRADING_DAYS = 252.0

# Fast bodies form aspects constantly, so a search that can choose its own
# body set needs to know which are which. Slow pairs are where the
# trickle-down premise is most plausible; the Sun alone produces an exact
# aspect with something every couple of weeks.
FAST_BODIES = ("MOON", "MERCURY", "SUN", "VENUS")
SLOW_BODIES = ("MARS", "JUPITER", "SATURN", "URANUS", "NEPTUNE", "PLUTO")

SIGNAL_RULES = ("level", "slope", "vs_mean")
OBJECTIVES = ("direction", "spearman", "sharpe")


@dataclass
class WaveParams:
    """One complete candidate configuration."""
    bodies: Tuple[str, ...] = ("MARS", "JUPITER", "SATURN", "URANUS")
    aspects: Tuple[float, ...] = (0.0, 90.0, 120.0, 180.0)
    orb: float = 2.0
    tau: float = 60.0
    horizon: int = 240
    lead: int = 0
    osc_period: Optional[float] = None
    osc_phase: float = 0.0
    rule: str = "level"
    smooth: int = 21
    signal_offset: int = 0
    threshold: float = 0.0

    def label(self) -> str:
        osc = f", osc {self.osc_period:.0f}d" if self.osc_period else ""
        return (f"{len(self.bodies)}b/{len(self.aspects)}a orb {self.orb:g} "
                f"tau {self.tau:.0f} reach {self.horizon}{osc} "
                f"{self.rule} sm{self.smooth} off{self.signal_offset:+d} "
                f"thr {self.threshold:.2f}")

    def to_row(self) -> Dict:
        d = asdict(self)
        d["bodies"] = " ".join(self.bodies)
        d["aspects"] = " ".join(f"{a:g}" for a in self.aspects)
        d["osc_period"] = self.osc_period or 0.0
        return d


# --------------------------------------------------------------------------
# Precomputation
# --------------------------------------------------------------------------
def precompute(
    start,
    end,
    bodies: Sequence[str],
    aspects: Sequence[float],
    max_orb: float = 8.0,
    frame: str = "geocentric",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute longitudes and the widest possible event set once.

    The search then filters this set per trial rather than recomputing the
    ephemeris hundreds of times. Filtering is exact rather than an
    approximation: event dates are local minima of |separation - aspect|,
    and where those minima fall does not depend on the orb. The orb only
    decides which minima are close enough to count, so selecting
    `offset <= orb` from a wide-orb set gives precisely the set a narrow-orb
    computation would have produced.
    """
    idx = A.calendar_index(start, end)
    lons = A.longitudes(idx, bodies=list(bodies), frame=frame)
    spec = A.EventSpec(aspects=list(aspects), orb_deg=float(max_orb),
                       min_separation_days=1)
    events = A.aspect_events(lons, spec)
    return lons, events


def filter_events(
    events: pd.DataFrame,
    params: WaveParams,
    min_separation_days: int = 3,
) -> pd.DataFrame:
    """Narrow the precomputed set to one candidate's bodies, aspects and orb."""
    if events.empty:
        return events

    want_bodies = set(params.bodies)
    keep = events["a"].isin(want_bodies) & events["b"].isin(want_bodies)

    asp = np.asarray(params.aspects, dtype=float)
    keep &= events["aspect"].apply(
        lambda v: bool(np.any(np.abs(asp - float(v)) < 0.05)))

    keep &= events["offset"] <= float(params.orb)
    out = events.loc[keep].copy()
    if out.empty:
        return out

    # Exactness weighting depends on the orb, so recompute it here rather
    # than inheriting the wide-orb value.
    exact = (1.0 - out["offset"] / max(float(params.orb), 1e-9)).clip(lower=0.05)
    pair_w = out.apply(lambda r: A.pair_weight(r["a"], r["b"]), axis=1)
    asp_w = out["aspect"].apply(lambda v: A.aspect_weight(float(v)))
    out["weight"] = pair_w * asp_w * exact

    # Re-apply the spacing rule after filtering, per pair and aspect.
    if min_separation_days > 1:
        out = out.sort_values("date")
        rows = []
        last: Dict[Tuple[str, float], pd.Timestamp] = {}
        for _, r in out.iterrows():
            key = (r["pair"], float(r["aspect"]))
            prev = last.get(key)
            if prev is None or (r["date"] - prev).days >= min_separation_days:
                rows.append(r)
                last[key] = r["date"]
        out = pd.DataFrame(rows) if rows else out.iloc[:0]

    return out.reset_index(drop=True)


def build_wave(index: pd.DatetimeIndex, events: pd.DataFrame,
               params: WaveParams) -> pd.Series:
    return W.composite_pressure(
        index, events,
        tau_days=params.tau, horizon_days=params.horizon,
        period_days=params.osc_period, phase_deg=params.osc_phase,
        lead_days=params.lead,
    )


# --------------------------------------------------------------------------
# Signal
# --------------------------------------------------------------------------
def signal_detail(wave: pd.Series, close_index: pd.DatetimeIndex,
                  params: WaveParams) -> pd.DataFrame:
    """
    Position plus the values that produced it.

    Returns a frame indexed by date with:
      position    +1 long, -1 short, 0 flat
      signal      the normalised value compared against the threshold
      wave_used   the raw wave value the signal read
      source_date the date that wave value is dated

    `source_date` exists because it is otherwise genuinely confusing. With a
    positive signal_offset, the position on a given day is driven by the wave
    several weeks later, so printing the wave value for the current date next
    to the position shows a number whose sign appears to contradict the
    position. Reporting the value actually used, and where it came from,
    makes the offset visible instead of mysterious.
    """
    w = wave.dropna()
    empty = pd.DataFrame({"position": 0.0, "signal": np.nan,
                          "wave_used": np.nan, "source_date": pd.NaT},
                         index=close_index)
    if w.empty:
        return empty

    off = int(params.signal_offset)
    # Shifting the wave's index back by `off` makes each date read the value
    # dated `off` days later. Only the wave is ever shifted; price is not.
    shifted = pd.Series(w.values, index=w.index - pd.Timedelta(off, "D")) if off else w

    smooth = max(int(params.smooth), 1)
    if params.rule == "slope":
        s = shifted.diff(smooth)
    elif params.rule == "vs_mean":
        s = shifted - shifted.rolling(smooth, min_periods=max(smooth // 2, 1)).mean()
    else:
        s = shifted

    def onto(series: pd.Series) -> pd.Series:
        return series.reindex(series.index.union(close_index)).ffill().reindex(close_index)

    s_on = onto(s)
    raw_on = onto(shifted)

    sd = float(np.nanstd(s_on.to_numpy(dtype=float)))
    s_norm = s_on / sd if sd > 1e-12 else s_on * 0.0

    thr = float(params.threshold)
    pos = pd.Series(0.0, index=close_index)
    pos[s_norm > thr] = 1.0
    pos[s_norm < -thr] = -1.0

    return pd.DataFrame({
        "position": pos.fillna(0.0),
        "signal": s_norm,
        "wave_used": raw_on,
        "source_date": pd.Series(close_index, index=close_index)
                       + pd.Timedelta(off, "D"),
    })


def build_signal(wave: pd.Series, close_index: pd.DatetimeIndex,
                 params: WaveParams) -> pd.Series:
    """
    Turn the wave into a position of +1, 0 or -1 on trading days.

    signal_offset shifts which wave value each day reads. Positive values
    read the wave AHEAD of the current date, which is legitimate because
    the wave is pure astronomy and computable in advance.
    """
    return signal_detail(wave, close_index, params)["position"]


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def score_window(
    close: pd.Series,
    position: pd.Series,
    objective: str = "direction",
    horizon: int = 5,
    tc_bps: float = 3.0,
) -> Dict[str, float]:
    """
    Score a position series against realised returns over one window.

    The position is lagged one bar before it meets a return, so nothing is
    ever scored against a price it could not have been acted on.
    """
    c = close.dropna()
    pos = position.reindex(c.index).fillna(0.0)
    if len(c) < 40:
        return {"score": np.nan, "n": len(c), "trades": 0,
                "hit_rate": np.nan, "sharpe": np.nan, "cum_return": np.nan,
                "exposure": np.nan}

    logp = np.log(c.to_numpy(dtype=float))
    tradable = pos.shift(1).fillna(0.0).to_numpy(dtype=float)

    r1 = np.concatenate([[0.0], np.diff(logp)])
    flips = np.abs(np.concatenate([[0.0], np.diff(tradable)]))
    net = tradable * r1 - flips * (float(tc_bps) / 1e4)

    equity = np.cumprod(1.0 + net)
    sd = float(np.std(net))
    sharpe = float(np.mean(net) / sd * np.sqrt(TRADING_DAYS)) if sd > 1e-12 else 0.0

    h = max(int(horizon), 1)
    if len(logp) > h:
        fwd = np.concatenate([logp[h:] - logp[:-h], np.full(h, np.nan)])
    else:
        fwd = np.full(len(logp), np.nan)

    live = (tradable != 0) & np.isfinite(fwd)
    hit = float(np.mean(np.sign(tradable[live]) == np.sign(fwd[live]))) if live.any() else np.nan

    if objective == "sharpe":
        score = sharpe
    elif objective == "spearman":
        ok = np.isfinite(fwd) & np.isfinite(tradable)
        if ok.sum() < 20:
            score = np.nan
        else:
            a = pd.Series(tradable[ok]).rank().to_numpy()
            b = pd.Series(fwd[ok]).rank().to_numpy()
            score = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 else np.nan
    else:
        score = hit

    return {
        "score": float(score) if score is not None and np.isfinite(score) else np.nan,
        "n": int(len(c)),
        "trades": int(np.sum(flips > 0)),
        "hit_rate": hit,
        "sharpe": sharpe,
        "cum_return": float(equity[-1] - 1.0),
        "exposure": float(np.mean(tradable != 0)),
    }


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------
def sample_params(rng: np.random.Generator,
                  body_pool: Sequence[str],
                  aspect_pool: Sequence[float]) -> WaveParams:
    """Draw one random candidate from sensible ranges."""
    n_bodies = int(rng.integers(2, min(len(body_pool), 7) + 1))
    bodies = tuple(rng.choice(list(body_pool), size=n_bodies, replace=False))

    n_asp = int(rng.integers(1, min(len(aspect_pool), 5) + 1))
    aspects = tuple(sorted(float(a) for a in
                           rng.choice(list(aspect_pool), size=n_asp, replace=False)))

    oscillate = bool(rng.random() < 0.45)
    return WaveParams(
        bodies=bodies,
        aspects=aspects,
        orb=float(rng.choice([1.0, 1.5, 2.0, 3.0, 4.0, 6.0])),
        tau=float(rng.choice([10, 21, 45, 90, 150, 250])),
        horizon=int(rng.choice([60, 120, 240, 400, 600])),
        lead=int(rng.choice([0, 0, 7, 21, 45])),
        osc_period=float(rng.choice([30, 60, 90, 180, 260, 360])) if oscillate else None,
        osc_phase=float(rng.choice([0, 45, 90, 135, 180, 225, 270, 315])) if oscillate else 0.0,
        rule=str(rng.choice(SIGNAL_RULES)),
        smooth=int(rng.choice([5, 10, 21, 45, 90])),
        signal_offset=int(rng.choice([-30, -10, 0, 0, 10, 30, 60])),
        threshold=float(rng.choice([0.0, 0.0, 0.25, 0.5, 0.75, 1.0])),
    )


def search(
    close: pd.Series,
    lons_index: pd.DatetimeIndex,
    events_all: pd.DataFrame,
    split_date,
    n_trials: int = 200,
    objective: str = "direction",
    horizon: int = 5,
    tc_bps: float = 3.0,
    body_pool: Optional[Sequence[str]] = None,
    aspect_pool: Optional[Sequence[float]] = None,
    min_separation_days: int = 3,
    min_trades: int = 6,
    min_exposure: float = 0.15,
    seed: int = 42,
    progress=None,
) -> pd.DataFrame:
    """
    Random search over wave parameters, scored on TRAIN, re-scored on TEST.

    Random rather than grid search on purpose: the parameter space here has
    around ten dimensions, so any grid fine enough to be interesting has
    more points than could ever be evaluated, and random sampling covers
    the space far better for a fixed budget.

    Candidates that trade too rarely or sit flat most of the time are
    rejected. Without that, the search reliably "wins" by finding a
    configuration that takes three trades in twelve years and happens to
    get them right.
    """
    close = close.dropna()
    split = pd.Timestamp(split_date)
    train_close = close.loc[close.index < split]
    test_close = close.loc[close.index >= split]

    if len(train_close) < 100 or len(test_close) < 60:
        raise RuntimeError(
            f"Split leaves {len(train_close)} train and {len(test_close)} test "
            f"bars. Need at least 100 and 60. Move the split date or widen "
            f"the overall range."
        )

    body_pool = list(body_pool) if body_pool else list(
        {b for b in events_all["a"]} | {b for b in events_all["b"]})
    aspect_pool = list(aspect_pool) if aspect_pool else sorted(
        set(float(v) for v in events_all["aspect"]))

    rng = np.random.default_rng(seed)
    rows: List[Dict] = []
    seen = set()

    for i in range(int(n_trials)):
        if progress:
            progress(i + 1, int(n_trials))
        p = sample_params(rng, body_pool, aspect_pool)

        key = (p.bodies, p.aspects, p.orb, p.tau, p.horizon, p.lead,
               p.osc_period, p.osc_phase, p.rule, p.smooth,
               p.signal_offset, p.threshold)
        if key in seen:
            continue
        seen.add(key)

        ev = filter_events(events_all, p, min_separation_days)
        if len(ev) < 4:
            continue

        wave = build_wave(lons_index, ev, p)
        pos = build_signal(wave, close.index, p)

        tr = score_window(train_close, pos, objective, horizon, tc_bps)
        if not np.isfinite(tr["score"]):
            continue
        if tr["trades"] < min_trades or tr["exposure"] < min_exposure:
            continue

        te = score_window(test_close, pos, objective, horizon, tc_bps)

        row = p.to_row()
        row.update({
            "n_events": len(ev),
            "train_score": round(tr["score"], 5),
            "test_score": round(te["score"], 5) if np.isfinite(te["score"]) else np.nan,
            "gap": (round(tr["score"] - te["score"], 5)
                    if np.isfinite(te["score"]) else np.nan),
            "train_sharpe": round(tr["sharpe"], 3),
            "test_sharpe": round(te["sharpe"], 3),
            "train_return": round(tr["cum_return"], 4),
            "test_return": round(te["cum_return"], 4),
            "test_trades": te["trades"],
            "exposure": round(tr["exposure"], 3),
            "label": p.label(),
        })
        rows.append(row)

    if not rows:
        raise RuntimeError(
            "No candidate passed the filters. Loosen min_trades or "
            "min_exposure, widen the body and aspect pools, or raise the "
            "trial count."
        )

    out = pd.DataFrame(rows).sort_values("train_score", ascending=False)
    return out.reset_index(drop=True)


def params_from_row(row) -> WaveParams:
    """Rebuild a WaveParams from a results row."""
    return WaveParams(
        bodies=tuple(str(row["bodies"]).split()),
        aspects=tuple(float(a) for a in str(row["aspects"]).split()),
        orb=float(row["orb"]),
        tau=float(row["tau"]),
        horizon=int(row["horizon"]),
        lead=int(row["lead"]),
        osc_period=float(row["osc_period"]) if float(row["osc_period"]) > 0 else None,
        osc_phase=float(row["osc_phase"]),
        rule=str(row["rule"]),
        smooth=int(row["smooth"]),
        signal_offset=int(row["signal_offset"]),
        threshold=float(row["threshold"]),
    )


# --------------------------------------------------------------------------
# Null model
# --------------------------------------------------------------------------
def null_distribution(
    close: pd.Series,
    lons_index: pd.DatetimeIndex,
    events_all: pd.DataFrame,
    split_date,
    n_trials: int = 60,
    objective: str = "direction",
    horizon: int = 5,
    tc_bps: float = 3.0,
    seed: int = 999,
    **kw,
) -> Dict[str, float]:
    """
    Run the identical search against a phase-randomised price series.

    This is the number that tells you what your search score means. The
    surrogate keeps your data's volatility and autocorrelation and destroys
    any real relationship, so the best score the search can reach on it is
    the best score the search reaches on nothing. If your real best is not
    clearly above this, the search found the shape of its own flexibility.
    """
    daily = W.to_calendar_daily(close.dropna())
    logp = np.log(daily.to_numpy(dtype=float))
    rng = np.random.default_rng(seed)
    surr = W._phase_randomise(W.detrend(logp, "linear"), rng)
    fake_daily = pd.Series(
        np.exp(surr - surr.mean() + float(np.log(close.mean()))),
        index=daily.index)
    fake = fake_daily.reindex(close.dropna().index).ffill().dropna()

    try:
        res = search(fake, lons_index, events_all, split_date,
                     n_trials=n_trials, objective=objective, horizon=horizon,
                     tc_bps=tc_bps, seed=seed, **kw)
    except Exception:
        return {"best_train": np.nan, "best_test": np.nan,
                "median_test": np.nan, "n": 0}

    return {
        "best_train": float(res["train_score"].max()),
        "best_test": float(res["test_score"].max()),
        "median_test": float(res["test_score"].median()),
        "n": int(len(res)),
    }


# --------------------------------------------------------------------------
# Forward projection
# --------------------------------------------------------------------------
def forward_plan(
    params: WaveParams,
    last_price_date,
    days_ahead: int = 180,
    frame: str = "geocentric",
    min_separation_days: int = 3,
    calendar_only: bool = True,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Apply calibrated parameters past the last price bar.

    Returns (wave, changes) where `changes` lists the dates the position
    flips, with the direction it flips to.

    Note what this is and is not. The wave is exact: it comes from orbital
    mechanics and does not depend on any market assumption. The position
    derived from it is only as good as the calibration, which is why the
    test-window score matters more than anything else on the screen.
    """
    last = pd.Timestamp(last_price_date).normalize()
    # Reach back far enough that decay tails from past events are already
    # accumulated when the projection window starts. Without this the wave
    # would begin from zero on the last price bar and ramp up artificially.
    back = int(max(params.horizon, params.tau * 4, 400))
    start = last - pd.Timedelta(back, "D")
    end = last + pd.Timedelta(int(days_ahead) + abs(params.signal_offset) + 30, "D")

    lons, events_all = precompute(start, end, params.bodies, params.aspects,
                                  max_orb=max(params.orb, 8.0), frame=frame)
    ev = filter_events(events_all, params, min_separation_days)
    wave = build_wave(lons.index, ev, params)

    future_index = pd.date_range(last, last + pd.Timedelta(int(days_ahead), "D"),
                                 freq="D" if calendar_only else "B")
    detail = signal_detail(wave, future_index, params)

    name = {1.0: "LONG", -1.0: "SHORT", 0.0: "FLAT"}
    changes = []
    prev = None
    for d, row in detail.iterrows():
        v = float(row["position"])
        if prev is None or v != prev:
            changes.append({
                "date": d.date(),
                "position": name[v],
                "signal": round(float(row["signal"]), 4)
                if np.isfinite(row["signal"]) else np.nan,
                "wave_used": round(float(row["wave_used"]), 4)
                if np.isfinite(row["wave_used"]) else np.nan,
                "driven_by_date": pd.Timestamp(row["source_date"]).date()
                if pd.notna(row["source_date"]) else None,
            })
            prev = v

    return wave, pd.DataFrame(changes)


def upcoming_drivers(params: WaveParams, last_price_date,
                     days_ahead: int = 180, frame: str = "geocentric",
                     min_separation_days: int = 3) -> pd.DataFrame:
    """The individual alignments falling inside the projection window."""
    last = pd.Timestamp(last_price_date).normalize()
    end = last + pd.Timedelta(int(days_ahead), "D")
    _, events_all = precompute(last, end, params.bodies, params.aspects,
                               max_orb=max(params.orb, 8.0), frame=frame)
    ev = filter_events(events_all, params, min_separation_days)
    if ev.empty:
        return ev
    ev = ev.sort_values("date")
    ev["date"] = pd.to_datetime(ev["date"]).dt.date
    return ev[["date", "pair", "aspect_name", "separation", "offset", "weight"]]

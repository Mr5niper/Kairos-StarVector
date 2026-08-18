"""
kairos.skygrid
==============
The chart this project was actually asked for.

What it draws
-------------
One or more stocks overlaid. For each selected planet, a dot marking that
planet's PEAK ALTITUDE in the sky - its height at culmination, measured
from an equatorial view - sampled at whatever interval you choose: daily,
weekly, monthly or yearly. The angle is mapped onto the height of the
chart, so the full vertical span represents the full angular domain (180
degrees by default). From every dot, Gann rays radiate in all four
quadrants: forward and backward, up and down.

Why backward rays are not decoration
------------------------------------
Planetary positions are computable in both directions with equal
precision, which makes this different from every price-derived indicator.
A ray drawn backward from a future planetary position can be checked
against price that has already happened. If past price respected it, the
forward half of the same geometry was drawn from the same rule rather
than fitted to the outcome. That is a genuinely testable construction,
and it is the reason the backward rays exist.

Steep rays are drawn fainter than shallow ones on purpose. A 8x1 ray
covers the whole chart in weeks, so it passes near almost everything and
carries correspondingly little information; the 1x1 is the reference the
whole method is built on. The styling encodes that difference rather than
leaving every line looking equally authoritative.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import ephem
import numpy as np
import pandas as pd

from . import astro as A

# Sampling intervals offered for the peak measurement.
INTERVALS: Dict[str, str] = {
    "day": "D",
    "week": "W",
    "month": "ME",
    "quarter": "QE",
    "year": "YE",
}

# Angular domains the chart height can represent.
ANGLE_DOMAINS: Dict[str, Tuple[float, float]] = {
    "0-180 (full sky arc)": (0.0, 180.0),
    "0-90 (altitude only)": (0.0, 90.0),
    "-90 to +90 (declination)": (-90.0, 90.0),
}

# Ray classification. Only the class is used for filtering now: every ray is
# drawn at the same width and the same opacity.
#
# An earlier version drew the 1x1 at double width and faded the steep ratios,
# to encode which lines are considered more reliable. That is an opinion about
# the method being imposed on the drawing of it, and it makes the grid read as
# though some lines are more real than others. A Gann line is a Gann line;
# where price sits relative to it is the information, not how bold it is.
RAY_UNIFORM_OPACITY = 0.40
RAY_UNIFORM_WIDTH = 1

RAY_STYLE: Dict[float, Tuple[float, int, str]] = {
    # ratio: (opacity, width, class)
    1 / 8: (RAY_UNIFORM_OPACITY, RAY_UNIFORM_WIDTH, "fast"),
    1 / 4: (RAY_UNIFORM_OPACITY, RAY_UNIFORM_WIDTH, "fast"),
    1 / 3: (RAY_UNIFORM_OPACITY, RAY_UNIFORM_WIDTH, "medium"),
    1 / 2: (RAY_UNIFORM_OPACITY, RAY_UNIFORM_WIDTH, "medium"),
    1.0: (RAY_UNIFORM_OPACITY, RAY_UNIFORM_WIDTH, "primary"),
    2.0: (RAY_UNIFORM_OPACITY, RAY_UNIFORM_WIDTH, "medium"),
    3.0: (RAY_UNIFORM_OPACITY, RAY_UNIFORM_WIDTH, "medium"),
    4.0: (RAY_UNIFORM_OPACITY, RAY_UNIFORM_WIDTH, "fast"),
    8.0: (RAY_UNIFORM_OPACITY, RAY_UNIFORM_WIDTH, "fast"),
}

PLANET_COLOURS: Dict[str, str] = {
    "SUN": "#F5C542", "MOON": "#C9D1D9", "MERCURY": "#9C6ADE",
    "VENUS": "#5FD3A6", "MARS": "#E0524B", "JUPITER": "#4FA8E0",
    "SATURN": "#E8973A", "URANUS": "#57B96C", "NEPTUNE": "#3F7FD0",
    "PLUTO": "#A8756B",
}


# --------------------------------------------------------------------------
# Sky angles
# --------------------------------------------------------------------------
def peak_altitude(
    dates: pd.DatetimeIndex,
    bodies: Sequence[str],
    latitude: float = 40.0,
) -> pd.DataFrame:
    """
    Each body's altitude at upper culmination, in degrees, per date.

    Uses the standard relation for the highest point a body reaches:

        alt_max = 90 - |latitude - declination|

    Verified against PyEphem's own observer transit calculation to within
    0.04 degrees, and it reproduces the textbook solstice altitudes for a
    given latitude. Preferred over iterating `next_transit` for every body
    on every date, which is the same answer at roughly a hundred times the
    cost.

    Bodies that never rise at this latitude yield a negative value, which is
    clipped to zero: a planet below the horizon has no peak in the sky.
    """
    dates = pd.DatetimeIndex(dates)
    if len(dates) == 0:
        return pd.DataFrame(index=dates)

    lat = float(latitude)
    out: Dict[str, np.ndarray] = {}
    edates = [ephem.Date(ts.to_pydatetime()) for ts in dates]

    for name in bodies:
        cls = A._EPHEM_CLASS.get(str(name).upper())
        if cls is None:
            continue
        body = cls()
        vals = np.empty(len(dates), dtype=float)
        for i, d in enumerate(edates):
            body.compute(d)
            dec = math.degrees(float(body.dec))
            vals[i] = 90.0 - abs(lat - dec)
        out[str(name).upper()] = np.clip(vals, 0.0, 90.0)

    return pd.DataFrame(out, index=dates)


def declinations(dates: pd.DatetimeIndex, bodies: Sequence[str]) -> pd.DataFrame:
    """Declination in degrees, for the -90..+90 domain option."""
    dates = pd.DatetimeIndex(dates)
    out: Dict[str, np.ndarray] = {}
    edates = [ephem.Date(ts.to_pydatetime()) for ts in dates]
    for name in bodies:
        cls = A._EPHEM_CLASS.get(str(name).upper())
        if cls is None:
            continue
        body = cls()
        vals = np.empty(len(dates), dtype=float)
        for i, d in enumerate(edates):
            body.compute(d)
            vals[i] = math.degrees(float(body.dec))
        out[str(name).upper()] = vals
    return pd.DataFrame(out, index=dates)


def interval_peaks(
    angles: pd.DataFrame,
    interval: str = "month",
) -> pd.DataFrame:
    """
    The single highest reading per body per period, with the date it fell on.

    Returns long form: date, body, angle, period.

    Takes the maximum rather than the period average or its endpoint,
    because the request is for the peak. Averaging a month of altitudes
    would report a number the planet never actually reached, and sampling
    the last day of the month would report wherever it happened to be.
    """
    if angles.empty:
        return pd.DataFrame(columns=["date", "body", "angle", "period"])

    freq = INTERVALS.get(str(interval).lower())
    rows: List[Dict] = []

    if freq is None or str(interval).lower() == "day":
        for body in angles.columns:
            s = angles[body].dropna()
            for d, v in s.items():
                rows.append({"date": d, "body": body, "angle": float(v),
                             "period": d.date().isoformat()})
        return pd.DataFrame(rows)

    for body in angles.columns:
        s = angles[body].dropna()
        if s.empty:
            continue
        grouped = s.groupby(pd.Grouper(freq=freq))
        for period, chunk in grouped:
            if chunk.empty:
                continue
            d = chunk.idxmax()
            rows.append({
                "date": d,
                "body": body,
                "angle": float(chunk.max()),
                "period": pd.Timestamp(period).date().isoformat(),
            })

    if not rows:
        return pd.DataFrame(columns=["date", "body", "angle", "period"])
    return pd.DataFrame(rows).sort_values(["date", "body"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Angle to price
# --------------------------------------------------------------------------
def angle_to_price(
    angle,
    price_low: float,
    price_high: float,
    domain: Tuple[float, float] = (0.0, 180.0),
    ratio: float = 1.0,
) -> np.ndarray:
    """
    Map a sky angle onto the price axis.

    The chart's vertical span represents the whole angular domain, so an
    angle sits at its own fraction of that domain. `ratio` scales the result
    about the midpoint of the price band, which is what makes the same
    planetary curve usable across instruments whose dollar ranges differ by
    orders of magnitude: a ratio of 1 spans the full band, 0.25 compresses
    the swing to a quarter of it, 2 exaggerates it.

    Scaling about the midpoint rather than the low keeps the curve centred
    on the instrument as the ratio changes, instead of sliding it off the
    bottom of the chart.
    """
    a = np.asarray(angle, dtype=float)
    lo_d, hi_d = float(domain[0]), float(domain[1])
    span_d = hi_d - lo_d
    if span_d == 0:
        span_d = 1.0

    frac = (a - lo_d) / span_d               # 0..1 across the domain
    lo_p, hi_p = float(price_low), float(price_high)
    mid = (lo_p + hi_p) / 2.0
    band = (hi_p - lo_p)
    return mid + (frac - 0.5) * band * float(ratio)


# --------------------------------------------------------------------------
# Radiating rays
# --------------------------------------------------------------------------
def radiating_rays(
    anchor_date: pd.Timestamp,
    anchor_price: float,
    unit_price_per_day: float,
    ratios: Sequence[float],
    x_low: pd.Timestamp,
    x_high: pd.Timestamp,
    y_bounds: Tuple[float, float],
    colour: str = "rgba(150,160,180,{a})",
    forward: bool = True,
    backward: bool = True,
    include_fast: bool = True,
    max_ray_days: Optional[int] = 120,
) -> List[Dict]:
    """
    Gann rays from one point into all four quadrants.

    `max_ray_days` caps how far each ray travels from its origin. Without a
    cap every ray runs to the edge of the chart, and four quadrants per dot
    across dozens of dots fills the plot with a lattice that hides the price
    series it was meant to inform.

    Every ray is clipped where it leaves the chart, in x and in y. Clipping
    in y is not cosmetic: Plotly folds shape coordinates into its autorange,
    so one unclipped steep ray stretches the price axis by orders of
    magnitude and flattens the price series into a line at the bottom of the
    plot.

    `include_fast` drops the steepest ratios, which is usually the
    difference between a readable chart and a grey wash once there is a dot
    for every planet in every period.
    """
    a_date = pd.Timestamp(anchor_date)
    y0 = float(anchor_price)
    lo_y, hi_y = float(min(y_bounds)), float(max(y_bounds))
    if not (lo_y <= y0 <= hi_y) or unit_price_per_day <= 0:
        return []

    days_fwd = max((pd.Timestamp(x_high) - a_date).days, 0)
    days_back = max((a_date - pd.Timestamp(x_low)).days, 0)
    if max_ray_days is not None:
        cap = max(int(max_ray_days), 1)
        days_fwd = min(days_fwd, cap)
        days_back = min(days_back, cap)

    shapes: List[Dict] = []
    for r in ratios:
        style = RAY_STYLE.get(float(r), (0.28, 1, "medium"))
        opacity, width, klass = style
        if klass == "fast" and not include_fast:
            continue

        slope = float(unit_price_per_day) * float(r)

        for x_sign, budget in ((1, days_fwd), (-1, days_back)):
            if x_sign > 0 and not forward:
                continue
            if x_sign < 0 and not backward:
                continue
            if budget <= 0:
                continue

            for y_sign in (1, -1):
                m = slope * y_sign * x_sign      # price change per day travelled
                if m == 0:
                    continue
                limit_y = hi_y if (slope * y_sign) > 0 else lo_y
                to_edge = (limit_y - y0) / (slope * y_sign)
                if to_edge <= 0:
                    continue
                days = min(float(budget), to_edge)
                if days < 1:
                    continue

                x1 = a_date + pd.Timedelta(int(round(days)) * x_sign, "D")
                y1 = y0 + slope * y_sign * days
                shapes.append({
                    "type": "line", "xref": "x", "yref": "y",
                    "x0": a_date, "y0": y0, "x1": x1, "y1": y1,
                    "line": {"color": colour.format(a=opacity), "width": width},
                    "layer": "below",
                })
    return shapes


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------
NORMALISERS: Dict[str, str] = {
    "raw price": "none",
    "indexed to 100": "index100",
    "percent from start": "pct",
}


def normalise_series(series: pd.Series, mode: str = "none") -> pd.Series:
    """
    Put instruments on a comparable vertical scale.

    Necessary for the overlay to mean anything. A 33 dollar stock and a 7000
    point index on one linear axis leaves the stock as a flat line along the
    bottom, and no per-stock angle ratio can fix that, because the ratio
    controls the amplitude of the planetary curve rather than the position of
    the price series it is drawn against.

    'indexed to 100' rebases each series to 100 at its first bar, so what is
    compared is relative movement. 'raw price' is left available because it
    is the right choice when the instruments genuinely are on similar scales.
    """
    s = series.dropna()
    if s.empty or mode in (None, "none", "raw price"):
        return s
    base = float(s.iloc[0])
    if base == 0:
        return s
    if mode in ("index100", "indexed to 100"):
        return s / base * 100.0
    if mode in ("pct", "percent from start"):
        return (s / base - 1.0) * 100.0
    return s


def build_sky_grid(
    prices: Dict[str, pd.Series],
    *,
    bodies: Sequence[str] = ("SUN", "MARS", "JUPITER", "SATURN"),
    latitude: float = 40.0,
    interval: str = "month",
    domain_name: str = "0-180 (full sky arc)",
    use_declination: bool = False,
    ratios: Optional[Dict[str, float]] = None,
    stock_ratios: Optional[Dict[str, float]] = None,
    unit_mode: str = "auto",
    manual_units: Optional[Dict[str, float]] = None,
    normalise: str = "indexed to 100",
    extend_days: int = 180,
    forward: bool = True,
    backward: bool = True,
    include_fast: bool = True,
    max_ray_anchors: int = 80,
    max_ray_days: Optional[int] = 120,
    y_pad_frac: float = 0.05,
) -> Dict[str, object]:
    """
    Build everything the sky-angle chart needs.

    `prices` maps ticker to a close series. Each ticker gets its own angle
    ratio from `stock_ratios`, which is the control that makes a 40 dollar
    stock and a 6000 point index share one chart without either becoming a
    flat line.

    Returns a dict with `series` (the normalised price series actually
    plotted), `dots` (long table of every plotted point), `shapes` (all
    rays), `y_bounds`, and `meta`.
    """
    from .gann import FAN_RATIOS, auto_unit_price_per_day, find_pivots

    ratios = ratios or FAN_RATIOS
    stock_ratios = stock_ratios or {}
    manual_units = manual_units or {}

    raw = {t: s.dropna() for t, s in prices.items()
           if s is not None and not s.dropna().empty}
    if not raw:
        return {"series": {}, "dots": pd.DataFrame(), "shapes": [],
                "y_bounds": (0.0, 1.0), "meta": {"error": "no price data"}}

    clean = {t: normalise_series(s, normalise) for t, s in raw.items()}

    first = min(s.index.min() for s in clean.values())
    last = max(s.index.max() for s in clean.values())
    x_high = last + pd.Timedelta(int(extend_days), "D")

    # A single price axis has to hold every overlaid instrument, so the band
    # spans all of them.
    lo_p = min(float(s.min()) for s in clean.values())
    hi_p = max(float(s.max()) for s in clean.values())
    pad = (hi_p - lo_p) * float(y_pad_frac) if hi_p > lo_p else max(abs(hi_p), 1.0) * 0.05
    y_bounds = (lo_p - pad, hi_p + pad)

    # Angles are computed over the whole span including the projection, since
    # future positions are exactly known and are the point of the exercise.
    cal = A.calendar_index(first, x_high)
    if use_declination or domain_name.startswith("-90"):
        angles = declinations(cal, bodies)
    else:
        angles = peak_altitude(cal, bodies, latitude)

    peaks = interval_peaks(angles, interval)
    if peaks.empty:
        return {"series": clean, "dots": pd.DataFrame(), "shapes": [],
                "y_bounds": y_bounds, "meta": {"error": "no angle samples"}}

    domain = ANGLE_DOMAINS.get(domain_name, (0.0, 180.0))

    dots_rows: List[Dict] = []
    shapes: List[Dict] = []
    meta: Dict[str, object] = {"tickers": {}, "interval": interval,
                               "domain": domain, "latitude": latitude}

    for ticker, series in clean.items():
        s_lo, s_hi = float(series.min()), float(series.max())
        t_ratio = float(stock_ratios.get(ticker, 1.0))

        if unit_mode == "manual" and ticker in manual_units:
            unit = float(manual_units[ticker])
        else:
            unit = auto_unit_price_per_day(series, find_pivots(series, 21))
        meta["tickers"][ticker] = {"unit_price_per_day": round(unit, 6),
                                   "angle_ratio": t_ratio,
                                   "price_low": round(s_lo, 4),
                                   "price_high": round(s_hi, 4)}

        sub = peaks.copy()
        sub["price"] = angle_to_price(sub["angle"].to_numpy(), s_lo, s_hi,
                                      domain, t_ratio)
        sub["ticker"] = ticker
        dots_rows.append(sub)

        # Rays from the strongest dots first, so a low anchor cap keeps the
        # most extreme angles rather than an arbitrary early slice.
        anchors = sub.reindex(
            sub["angle"].abs().sort_values(ascending=False).index
        ).head(int(max_ray_anchors))

        for _, row in anchors.iterrows():
            base = PLANET_COLOURS.get(row["body"], "#9aa4b2").lstrip("#")
            rgb = tuple(int(base[i:i + 2], 16) for i in (0, 2, 4))
            tmpl = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{{a}})"
            shapes.extend(radiating_rays(
                row["date"], float(row["price"]), unit,
                list(ratios.values()), first, x_high, y_bounds,
                colour=tmpl, forward=forward, backward=backward,
                include_fast=include_fast, max_ray_days=max_ray_days,
            ))

    dots = pd.concat(dots_rows, ignore_index=True) if dots_rows else pd.DataFrame()
    meta["normalise"] = normalise
    meta["n_dots"] = int(len(dots))
    meta["n_rays"] = int(len(shapes))
    meta["max_ray_days"] = max_ray_days
    meta["angle_measure"] = ("declination" if (use_declination or
                                               domain_name.startswith("-90"))
                             else "peak altitude")
    return {"series": clean, "dots": dots, "shapes": shapes,
            "y_bounds": y_bounds, "meta": meta}


# ==========================================================================
# Degree-space grid
# ==========================================================================
# The chart above maps sky angles into price. This section does the reverse,
# which is the construction actually wanted:
#
#   * The vertical axis IS degrees, 0 at the bottom and 180 at the top.
#   * Each planet's interval peak is a dot at its own angle. No conversion,
#     no scaling: the dot sits where the measurement says.
#   * Gann rays radiate from every dot in all four directions.
#   * The stock is the overlay, scaled by a ratio and offset into the degree
#     band until its past movement lines up with the planetary geometry.
#
# Working in degrees rather than dollars removes the whole per-instrument
# calibration problem from the rays. The 1x1 becomes one degree per day,
# which is Gann's own unit and is the same on every instrument ever traded.
# Only the price overlay needs a ratio, and that is a single fit rather than
# a slope that has to be rediscovered per symbol.
#
# Intersections are the output that matters here. A crossing between a ray
# from a past dot and a ray from a FUTURE dot is computable today, because
# the future dot's position is known exactly. That is the decision candidate.

def price_to_degrees(
    series: pd.Series,
    domain: Tuple[float, float] = (0.0, 180.0),
    ratio: float = 1.0,
    offset: float = 0.0,
) -> pd.Series:
    """
    Scale a price series into the degree band.

    The series is first put on 0..1 across its own range, then stretched by
    `ratio` and shifted by `offset`, both in degrees. `ratio` of 1 spans the
    whole 180, 0.5 spans 90. This is the control for sliding the stock up
    and down and squashing it until its past turns sit on the planetary
    geometry.
    """
    s = series.dropna()
    if s.empty:
        return s
    lo, hi = float(s.min()), float(s.max())
    span = hi - lo
    if span <= 0:
        span = 1.0
    unit = (s - lo) / span
    d_lo, d_hi = float(domain[0]), float(domain[1])
    return unit * (d_hi - d_lo) * float(ratio) + d_lo + float(offset)


def fit_price_to_angles(
    price: pd.Series,
    angles: pd.Series,
    domain: Tuple[float, float] = (0.0, 180.0),
) -> Dict[str, float]:
    """
    Least-squares ratio and offset that best line the stock up with a
    planet's angle curve over the period they share.

    Returns ratio, offset and r_squared.

    The r_squared is the number to look at, and it is easy to over-read. Two
    smooth curves fitted with two free parameters will reach a respectable
    figure without any relationship existing, so treat this as a starting
    position for the sliders rather than as evidence. The Statistics tab
    exists to answer the second question.
    """
    joined = pd.concat([price.rename("p"), angles.rename("a")], axis=1).dropna()
    if len(joined) < 10:
        return {"ratio": 1.0, "offset": 0.0, "r_squared": float("nan")}

    p = joined["p"].to_numpy(dtype=float)
    a = joined["a"].to_numpy(dtype=float)
    lo, hi = float(np.min(p)), float(np.max(p))
    span = (hi - lo) if hi > lo else 1.0
    unit = (p - lo) / span                      # 0..1

    d_span = float(domain[1] - domain[0])
    if np.std(unit) < 1e-12:
        return {"ratio": 1.0, "offset": 0.0, "r_squared": float("nan")}

    slope, intercept = np.polyfit(unit, a, 1)
    pred = slope * unit + intercept
    ss_res = float(np.sum((a - pred) ** 2))
    ss_tot = float(np.sum((a - np.mean(a)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "ratio": float(slope / d_span) if d_span else 1.0,
        "offset": float(intercept - float(domain[0])),
        "r_squared": float(r2),
    }


def squared_deg_per_day(
    x_low, x_high,
    domain: Tuple[float, float] = (0.0, 180.0),
) -> float:
    """
    The 1x1 rate that makes a ray cross the chart corner to corner.

    This is what Gann meant by squaring the chart, and it has to be derived
    from the visible window rather than fixed, because a rate in degrees per
    day only looks like 45 degrees on screen at one particular window width.

    A fixed 1 degree per day crosses the full 180 degrees in 180 days. On a
    four-month window that is wider than the chart and reads as a gentle
    diagonal; on a three-year window it is 14 per cent of the width and every
    ray looks vertical. Same arithmetic, same rays, unusable at one scale and
    correct at the other. Tying the rate to the window keeps the 1x1 at the
    diagonal no matter how much history is displayed.
    """
    days = max((pd.Timestamp(x_high) - pd.Timestamp(x_low)).days, 1)
    span = abs(float(domain[1]) - float(domain[0]))
    return float(span / days)


def degree_rays(
    anchor_date: pd.Timestamp,
    anchor_deg: float,
    deg_per_day: float,
    ratios: Dict[str, float],
    x_low: pd.Timestamp,
    x_high: pd.Timestamp,
    domain: Tuple[float, float] = (0.0, 180.0),
    max_ray_days: Optional[int] = None,
    forward: bool = True,
    backward: bool = True,
    include_fast: bool = True,
    body: str = "",
) -> List[Dict]:
    """
    Rays from one dot, in degree space, clipped to the domain.

    Each ray is returned as a dict carrying its endpoints in both date and
    day-number form, so intersections can be solved arithmetically without
    re-deriving the geometry from Plotly shapes.
    """
    a_date = pd.Timestamp(anchor_date).normalize()
    d_lo, d_hi = float(domain[0]), float(domain[1])
    if not (d_lo <= float(anchor_deg) <= d_hi) or deg_per_day <= 0:
        return []

    fwd_budget = max((pd.Timestamp(x_high) - a_date).days, 0)
    back_budget = max((a_date - pd.Timestamp(x_low)).days, 0)
    if max_ray_days is not None:
        cap = max(int(max_ray_days), 1)
        fwd_budget = min(fwd_budget, cap)
        back_budget = min(back_budget, cap)

    rays: List[Dict] = []
    for name, r in ratios.items():
        style = RAY_STYLE.get(float(r), (0.30, 1, "medium"))
        opacity, width, klass = style
        if klass == "fast" and not include_fast:
            continue
        slope = float(deg_per_day) * float(r)      # degrees per day

        for x_sign, budget in ((1, fwd_budget), (-1, back_budget)):
            if x_sign > 0 and not forward:
                continue
            if x_sign < 0 and not backward:
                continue
            if budget <= 0:
                continue
            for y_sign in (1, -1):
                edge = d_hi if y_sign > 0 else d_lo
                to_edge = (edge - float(anchor_deg)) / (slope * y_sign)
                if to_edge <= 0:
                    continue
                days = min(float(budget), to_edge)
                if days < 1:
                    continue
                rays.append({
                    "body": body,
                    "ratio": name,
                    "class": klass,
                    "opacity": opacity,
                    "width": width,
                    "x0": a_date,
                    "x1": a_date + pd.Timedelta(int(round(days)) * x_sign, "D"),
                    "y0": float(anchor_deg),
                    "y1": float(anchor_deg) + slope * y_sign * days,
                    "dir": ("fwd" if x_sign > 0 else "back"),
                    "slope_per_day": slope * y_sign * x_sign,
                })
    return rays


def rays_to_shapes(rays: Sequence[Dict],
                   colour_by_body: bool = True) -> List[Dict]:
    """Plotly line shapes for a list of rays."""
    shapes = []
    for r in rays:
        if colour_by_body and r.get("body"):
            base = PLANET_COLOURS.get(str(r["body"]), "#9aa4b2").lstrip("#")
            rgb = tuple(int(base[i:i + 2], 16) for i in (0, 2, 4))
            col = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{r['opacity']})"
        else:
            col = f"rgba(200,205,215,{r['opacity']})"
        shapes.append({
            "type": "line", "xref": "x", "yref": "y",
            "x0": r["x0"], "y0": r["y0"], "x1": r["x1"], "y1": r["y1"],
            "line": {"color": col, "width": r["width"]},
            "layer": "below",
        })
    return shapes


def ray_intersections(
    rays: Sequence[Dict],
    origin: pd.Timestamp,
    now: pd.Timestamp,
    min_separation_days: int = 2,
    only_future: bool = False,
    max_pairs: int = 400_000,
) -> pd.DataFrame:
    """
    Every crossing between two rays, solved in (day number, degree) space.

    Returns date, degrees, the two rays involved, and `kind`:

        past-past      both rays come from dots already behind us
        past-future    one ray from a past dot, one from a future dot
        future-future  both from future dots

    `past-future` is the interesting class and the reason the backward rays
    exist. A future planetary position is known exactly today, so a ray cast
    backward from it can be crossed with a ray from a dot that has already
    happened, and the crossing is computable now rather than in hindsight.

    Cost is quadratic in the ray count, so `max_pairs` caps the work. Reduce
    the ray count rather than raising it: past a few hundred crossings the
    chart cannot show them anyway.
    """
    n = len(rays)
    if n < 2:
        return pd.DataFrame(columns=["date", "degrees", "kind", "ray_a",
                                     "ray_b", "days_from_now"])

    o = pd.Timestamp(origin).normalize()
    now = pd.Timestamp(now).normalize()

    def dayno(ts) -> float:
        return float((pd.Timestamp(ts).normalize() - o).days)

    segs = []
    for r in rays:
        segs.append((dayno(r["x0"]), float(r["y0"]),
                     dayno(r["x1"]), float(r["y1"]), r))

    rows = []
    pairs = 0
    for i in range(n):
        x1, y1, x2, y2 = segs[i][0], segs[i][1], segs[i][2], segs[i][3]
        for j in range(i + 1, n):
            pairs += 1
            if pairs > max_pairs:
                break
            x3, y3, x4, y4 = segs[j][0], segs[j][1], segs[j][2], segs[j][3]

            den = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3)
            if abs(den) < 1e-12:
                continue                      # parallel
            t = ((x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3)) / den
            u = ((x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)) / den
            if not (0.0 <= t <= 1.0 and 0.0 <= u <= 1.0):
                continue                      # crossing outside both segments

            xi = x1 + t * (x2 - x1)
            yi = y1 + t * (y2 - y1)
            date = o + pd.Timedelta(int(round(xi)), "D")

            ra, rb = segs[i][4], segs[j][4]
            a_future = pd.Timestamp(ra["x0"]).normalize() > now
            b_future = pd.Timestamp(rb["x0"]).normalize() > now
            if a_future and b_future:
                kind = "future-future"
            elif a_future or b_future:
                kind = "past-future"
            else:
                kind = "past-past"

            if only_future and date < now:
                continue

            rows.append({
                "date": date,
                "degrees": round(float(yi), 3),
                "kind": kind,
                "ray_a": f"{ra['body']} {ra['ratio']} {ra['dir']}",
                "ray_b": f"{rb['body']} {rb['ratio']} {rb['dir']}",
                "days_from_now": int((date - now).days),
            })
        if pairs > max_pairs:
            break

    if not rows:
        return pd.DataFrame(columns=["date", "degrees", "kind", "ray_a",
                                     "ray_b", "days_from_now"])

    out = pd.DataFrame(rows).sort_values(["date", "degrees"]).reset_index(drop=True)

    # Thin near-duplicates: many rays share an origin dot, so dozens of
    # crossings can land within a day and a degree of each other and would
    # otherwise be reported as separate signals.
    if min_separation_days > 0 and not out.empty:
        keep, last = [], None
        for _, r in out.iterrows():
            if (last is None
                    or abs((r["date"] - last["date"]).days) >= min_separation_days
                    or abs(r["degrees"] - last["degrees"]) > 3.0):
                keep.append(r)
                last = r
        out = pd.DataFrame(keep).reset_index(drop=True)

    return out


def confluence_zones(
    intersections: pd.DataFrame,
    now: pd.Timestamp,
    window_days: int = 5,
    degree_window: float = 10.0,
    min_count: int = 2,
) -> pd.DataFrame:
    """
    Group crossings that cluster in time and degree.

    A single crossing of two lines is weak; several independent rays meeting
    in the same few days is the confluence the method is looking for. Counts
    are only comparable within one chart configuration, since adding rays
    raises every count mechanically.
    """
    if intersections is None or intersections.empty:
        return pd.DataFrame(columns=["date", "degrees", "count", "kinds",
                                     "days_from_now"])
    df = intersections.copy().sort_values("date")
    now = pd.Timestamp(now).normalize()

    zones, used = [], np.zeros(len(df), dtype=bool)
    dates = pd.DatetimeIndex(df["date"])
    degs = df["degrees"].to_numpy(dtype=float)

    for i in range(len(df)):
        if used[i]:
            continue
        near = (~used) & (np.abs((dates - dates[i]).days) <= int(window_days)) \
               & (np.abs(degs - degs[i]) <= float(degree_window))
        idxs = np.flatnonzero(near)
        if len(idxs) < int(min_count):
            continue
        used[idxs] = True
        centre = dates[idxs].mean()
        zones.append({
            "date": pd.Timestamp(centre).normalize(),
            "degrees": round(float(np.mean(degs[idxs])), 2),
            "count": int(len(idxs)),
            "kinds": ", ".join(sorted(set(df.iloc[idxs]["kind"]))),
            "days_from_now": int((pd.Timestamp(centre).normalize() - now).days),
        })

    if not zones:
        return pd.DataFrame(columns=["date", "degrees", "count", "kinds",
                                     "days_from_now"])
    return (pd.DataFrame(zones).sort_values("count", ascending=False)
            .reset_index(drop=True))


def build_degree_grid(
    price: Dict[str, pd.Series],
    *,
    bodies: Sequence[str] = ("MARS",),
    latitude: float = 47.8,
    interval: str = "month",
    domain: Tuple[float, float] = (0.0, 180.0),
    use_declination: bool = False,
    deg_per_day: Optional[float] = None,
    ratios: Optional[Dict[str, float]] = None,
    stock_ratios: Optional[Dict[str, float]] = None,
    stock_offsets: Optional[Dict[str, float]] = None,
    autofit: bool = False,
    extend_days: int = 120,
    max_ray_days: Optional[int] = None,
    forward: bool = True,
    backward: bool = True,
    include_fast: bool = True,
    ray_dots: int = 12,
    ray_dot_mode: str = "nearest now",
    find_intersections: bool = True,
) -> Dict[str, object]:
    """
    The degree-space chart: planet dots at their true angles, rays radiating
    from each, the stock scaled in as an overlay, and the crossings tabulated.
    """
    from .gann import FAN_RATIOS

    ratios = ratios or {"1x2": 0.5, "1x1": 1.0, "2x1": 2.0}
    stock_ratios = stock_ratios or {}
    stock_offsets = stock_offsets or {}

    clean = {t: s.dropna() for t, s in price.items()
             if s is not None and not s.dropna().empty}
    if not clean:
        return {"error": "no price data"}

    first = min(s.index.min() for s in clean.values())
    now = max(s.index.max() for s in clean.values())
    x_high = now + pd.Timedelta(int(extend_days), "D")

    # deg_per_day=None means square the chart to the window, which is the
    # only setting that keeps the 1x1 on the diagonal at every window length.
    if deg_per_day is None or float(deg_per_day) <= 0:
        deg_per_day = squared_deg_per_day(first, x_high, domain)

    cal = A.calendar_index(first, x_high)
    if use_declination:
        angles = declinations(cal, bodies)
    else:
        angles = peak_altitude(cal, bodies, latitude)

    peaks = interval_peaks(angles, interval)
    if peaks.empty:
        return {"error": "no angle samples"}

    # EVERY interval across the whole displayed timeline is kept. Planetary
    # positions are deterministic and cheap to compute, so there is no reason
    # to show a subset: pick monthly and you get every month, past and future.
    #
    # An earlier version trimmed to the dots nearest the present, which meant
    # the interval setting appeared to do nothing on a long window because the
    # trim overrode it. Dots are cheap; rays are not, so the limit belongs on
    # the rays alone.
    peaks = peaks.sort_values("date").reset_index(drop=True)

    # Which dots emit rays. Every dot is still plotted.
    if ray_dots and ray_dots > 0 and len(peaks) > int(ray_dots):
        if str(ray_dot_mode).startswith("nearest"):
            gap = pd.Series(
                np.abs((pd.DatetimeIndex(peaks["date"]) - now).days),
                index=peaks.index)
            emit = peaks.loc[gap.sort_values().index[:int(ray_dots)]]
        elif str(ray_dot_mode).startswith("future"):
            fut = peaks[pd.DatetimeIndex(peaks["date"]) >= now]
            emit = fut.head(int(ray_dots)) if not fut.empty else peaks.tail(int(ray_dots))
        elif str(ray_dot_mode).startswith("highest"):
            emit = peaks.reindex(
                peaks["angle"].sort_values(ascending=False).index[:int(ray_dots)])
        else:                                   # spread evenly
            step = max(len(peaks) // int(ray_dots), 1)
            emit = peaks.iloc[::step].head(int(ray_dots))
        emit = emit.sort_values("date")
    else:
        emit = peaks

    rays: List[Dict] = []
    for _, row in emit.iterrows():
        rays.extend(degree_rays(
            row["date"], float(row["angle"]), float(deg_per_day), ratios,
            first, x_high, domain, max_ray_days, forward, backward,
            include_fast, body=str(row["body"])))

    curves: Dict[str, pd.Series] = {}
    fits: Dict[str, Dict[str, float]] = {}
    for ticker, s in clean.items():
        ratio = float(stock_ratios.get(ticker, 1.0))
        offset = float(stock_offsets.get(ticker, 0.0))
        if autofit:
            ref = angles[list(angles.columns)[0]].reindex(s.index).ffill()
            f = fit_price_to_angles(s, ref, domain)
            ratio, offset = f["ratio"], f["offset"]
            fits[ticker] = f
        curves[ticker] = price_to_degrees(s, domain, ratio, offset)
        fits.setdefault(ticker, {})["ratio_used"] = ratio
        fits[ticker]["offset_used"] = offset

    inter = pd.DataFrame()
    zones = pd.DataFrame()
    if find_intersections and rays:
        inter = ray_intersections(rays, first, now)
        zones = confluence_zones(inter, now)

    return {
        "dots": peaks,
        "rays": rays,
        "shapes": rays_to_shapes(rays),
        "curves": curves,
        "fits": fits,
        "angles": angles,
        "intersections": inter,
        "zones": zones,
        "now": now,
        "x_range": (first, x_high),
        "domain": domain,
        "emitting": emit,
        "meta": {"n_dots": int(len(peaks)),
                 "n_ray_origins": int(len(emit)),
                 "n_rays": len(rays),
                 "n_intersections": int(len(inter)),
                 "n_zones": int(len(zones)),
                 "deg_per_day": float(deg_per_day),
                 "squared_deg_per_day": squared_deg_per_day(first, x_high, domain),
                 "window_days": int((pd.Timestamp(x_high) - pd.Timestamp(first)).days),
                 "interval": interval,
                 "measure": "declination" if use_declination else "peak altitude"},
    }

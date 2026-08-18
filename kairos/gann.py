"""
kairos.gann
===========
Gann geometry, and the bridge between planetary longitude and price.

What changed from the original build
-----------------------------------
The old `gann_grid.py` had two structural problems:

  * The "1x1" fan slope was derived as (price range) / (total days) of
    whatever window happened to be loaded. That makes the angle depend on
    your date picker rather than on the market, so the same chart drew
    different fans at different zoom levels. Gann's 1x1 is a fixed rate
    of price per unit of time and has to be anchored to something stable.

  * Fans were anchored only on planetary alignment dates. Gann anchored
    on pivots: significant highs and lows. Alignment-anchored fans are
    worth drawing too, but on their own they are unanchored to price
    structure and will land mid-trend at arbitrary levels.

Both are fixed here. Pivots are detected from the price series, the 1x1
unit is either explicit or derived from a stated scale, and alignment
anchoring is an additional option rather than the only one.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Gann's preferred fan ratios: price units per time unit.
FAN_RATIOS: Dict[str, float] = {
    "1x8": 1 / 8, "1x4": 1 / 4, "1x3": 1 / 3, "1x2": 1 / 2,
    "1x1": 1.0,
    "2x1": 2.0, "3x1": 3.0, "4x1": 4.0, "8x1": 8.0,
}

# Divisions of the circle Gann used for time counts.
TIME_DEGREES: List[float] = [30, 45, 60, 72, 90, 120, 135, 144, 180, 225, 270, 315, 360]

SOLAR_YEAR_DAYS = 365.2422


# --------------------------------------------------------------------------
# Pivots
# --------------------------------------------------------------------------
def find_pivots(
    close: pd.Series,
    window: int = 21,
    min_separation: int = 10,
) -> pd.DataFrame:
    """
    Swing highs and lows: bars that are the extreme of a centred window.

    `window` is the half-life of your definition of "significant". A
    21-bar window finds monthly swings; 63 finds quarterly ones. Anchor
    your fans on these rather than on arbitrary dates.
    """
    s = close.dropna()
    if len(s) < window * 2 + 1:
        return pd.DataFrame(columns=["date", "price", "kind", "index"])

    v = s.to_numpy(dtype=float)
    n = len(v)
    w = int(window)
    rows = []

    for i in range(w, n - w):
        seg = v[i - w:i + w + 1]
        if v[i] == seg.max():
            rows.append({"date": s.index[i], "price": float(v[i]), "kind": "high", "index": i})
        elif v[i] == seg.min():
            rows.append({"date": s.index[i], "price": float(v[i]), "kind": "low", "index": i})

    if not rows:
        return pd.DataFrame(columns=["date", "price", "kind", "index"])

    df = pd.DataFrame(rows).sort_values("index").reset_index(drop=True)

    # Thin out clusters, keeping the more extreme member of each cluster.
    keep, last = [], None
    for _, r in df.iterrows():
        if last is None or (r["index"] - last["index"]) >= min_separation or r["kind"] != last["kind"]:
            keep.append(r)
            last = r
        else:
            better = (r["price"] > last["price"]) if r["kind"] == "high" else (r["price"] < last["price"])
            if better:
                keep[-1] = r
                last = r
    return pd.DataFrame(keep).reset_index(drop=True)


# --------------------------------------------------------------------------
# Square of nine
# --------------------------------------------------------------------------
def square_of_nine_levels(
    base_price: float,
    degrees: Sequence[float] = (45, 90, 135, 180, 225, 270, 315, 360),
    rotations: int = 2,
    both_directions: bool = True,
) -> pd.DataFrame:
    """
    Gann's Square of Nine as price levels.

    The construction: one full rotation of the spiral adds 2 to the square
    root of the price, so a rotation of `deg` degrees from a base gives

        price = (sqrt(base) + 2 * deg / 360) ** 2

    Going down uses the same formula with a minus sign. `rotations`
    extends past 360 degrees for further-out levels.
    """
    base = float(base_price)
    if base <= 0:
        return pd.DataFrame(columns=["degrees", "rotation", "direction", "price"])

    root = math.sqrt(base)
    rows = []
    for rot in range(int(rotations)):
        for deg in degrees:
            total = rot * 360.0 + float(deg)
            for sgn, name in ((1, "up"), (-1, "down")) if both_directions else ((1, "up"),):
                r = root + sgn * 2.0 * total / 360.0
                if r <= 0:
                    continue
                rows.append({
                    "degrees": float(total),
                    "rotation": rot + 1,
                    "direction": name,
                    "price": float(r * r),
                })
    df = pd.DataFrame(rows)
    return df.sort_values("price").reset_index(drop=True) if not df.empty else df


def price_to_sq9_degrees(price: float, base_price: float) -> float:
    """Inverse of the above: where a price sits on the spiral, in degrees."""
    if price <= 0 or base_price <= 0:
        return float("nan")
    return (math.sqrt(price) - math.sqrt(base_price)) * 180.0


# --------------------------------------------------------------------------
# Fans
# --------------------------------------------------------------------------
def auto_unit_price_per_day(
    close: pd.Series,
    pivots: Optional[pd.DataFrame] = None,
    lookback_days: Optional[int] = 730,
) -> float:
    """
    A stable 1x1 rate: the median price change per day across swings.

    Measured pivot to pivot, not bar to bar. Median absolute DAILY change
    was the first attempt and it is far too steep: daily noise on this data
    gives 0.65 per day, which projects a 1x1 line to 1315 on an instrument
    that never traded above 100. Trend rate and noise rate are different
    quantities, and the fan needs the former.

    Deliberately not (range / days in view), which was the original build's
    approach: that makes the angle depend on the date picker, so the same
    market drew different fans at different zoom levels.

    If no pivots are supplied they are detected here.
    """
    s = close.dropna()
    if len(s) < 30:
        return 1.0
    if lookback_days:
        cutoff = s.index.max() - pd.Timedelta(int(lookback_days), "D")
        seg = s.loc[s.index >= cutoff]
        if len(seg) >= 60:
            s = seg

    pv = pivots if pivots is not None else find_pivots(s, window=21)
    if pv is not None and len(pv) >= 3:
        pv = pv.sort_values("date")
        rates = []
        prices = pv["price"].to_numpy(dtype=float)
        dates = pd.DatetimeIndex(pv["date"])
        for i in range(1, len(pv)):
            days = (dates[i] - dates[i - 1]).days
            if days > 0:
                rates.append(abs(prices[i] - prices[i - 1]) / days)
        if rates:
            unit = float(np.median(rates))
            if unit > 0:
                return unit

    d = np.abs(np.diff(s.to_numpy(dtype=float)))
    unit = float(np.median(d[d > 0])) if np.any(d > 0) else 1.0
    return unit if unit > 0 else 1.0


def fan_horizon_days(close: pd.Series, unit_price_per_day: float,
                     max_multiple: float = 2.0) -> int:
    """
    How far a fan stays on the chart before its lines leave it.

    A 1x1 line rises without limit while price does not, so past a certain
    distance from the anchor every ratio sits above the highest price ever
    traded and the whole fan reads "price is below everything". That is not
    a weak market, it is an expired fan. Analysis beyond this horizon is
    meaningless and is masked out rather than reported.

    Returns the days after which the 1x1 exceeds `max_multiple` times the
    observed price range.
    """
    s = close.dropna()
    if s.empty or unit_price_per_day <= 0:
        return 365
    span = float(s.max() - s.min())
    if span <= 0:
        span = float(s.mean()) * 0.1
    return int(max(30, min(span * float(max_multiple) / float(unit_price_per_day), 4000)))


def fan_lines(
    anchor_date: pd.Timestamp,
    anchor_price: float,
    unit_price_per_day: float,
    ratios: Sequence[float],
    extend_to: pd.Timestamp,
    both_directions: bool = True,
    colour: str = "rgba(150,160,180,0.45)",
    width: int = 1,
    y_bounds: Optional[Tuple[float, float]] = None,
) -> List[Dict]:
    """
    Plotly line shapes for a Gann fan from one anchor.

    A 1x1 line rises `unit_price_per_day` per calendar day. Ratio 2x1
    rises twice as fast, 1x2 half as fast.

    `y_bounds` truncates each ray where it leaves the visible price band,
    and supplying it is close to mandatory. A steep ratio does not stop: an
    8x1 line at 40 points per day, anchored 1300 bars back and extended 180
    days, ends at 474,000. Plotly includes shape coordinates in its
    autorange calculation, so a single such ray stretches the price axis to
    cover it and crushes the candles into a flat line at the bottom of the
    chart. Clipping the ray at the edge of the band draws exactly the same
    visible line while leaving the axis alone.
    """
    a_date = pd.Timestamp(anchor_date)
    end = pd.Timestamp(extend_to)
    total_days = (end - a_date).days
    if total_days <= 0 or unit_price_per_day <= 0:
        return []

    shapes = []
    for r in ratios:
        slope = float(unit_price_per_day) * float(r)      # price per day
        for sgn in ((1, -1) if both_directions else (1,)):
            m = slope * sgn
            days = float(total_days)

            if y_bounds is not None and m != 0.0:
                lo, hi = float(min(y_bounds)), float(max(y_bounds))
                limit = hi if m > 0 else lo
                to_edge = (limit - float(anchor_price)) / m
                if to_edge <= 0:
                    # The anchor already sits outside the band in this
                    # direction, so the ray is never visible.
                    continue
                days = min(days, to_edge)

            if days <= 0:
                continue

            x1 = a_date + pd.Timedelta(int(round(days)), "D")
            y1 = float(anchor_price) + m * days
            shapes.append({
                "type": "line", "xref": "x", "yref": "y",
                "x0": a_date, "y0": float(anchor_price),
                "x1": x1, "y1": y1,
                "line": {"color": colour, "width": width},
                "layer": "below",
            })
    return shapes


def horizontal_levels(
    prices: Iterable[float],
    x0: pd.Timestamp,
    x1: pd.Timestamp,
    colour: str = "rgba(80,120,210,0.22)",
    width: int = 1,
    dash: Optional[str] = None,
) -> List[Dict]:
    shapes = []
    for p in prices:
        if not np.isfinite(p):
            continue
        line = {"color": colour, "width": width}
        if dash:
            line["dash"] = dash
        shapes.append({
            "type": "line", "xref": "x", "yref": "y",
            "x0": pd.Timestamp(x0), "y0": float(p),
            "x1": pd.Timestamp(x1), "y1": float(p),
            "line": line, "layer": "below",
        })
    return shapes


def round_number_levels(low: float, high: float, step: float) -> List[float]:
    if step <= 0 or not np.isfinite(low) or not np.isfinite(high):
        return []
    start = math.floor(low / step) * step
    out, v, guard = [], start, 0
    while v <= high and guard < 5000:
        out.append(float(v))
        v += step
        guard += 1
    return out


# --------------------------------------------------------------------------
# Time cycles
# --------------------------------------------------------------------------
def time_cycle_dates(
    anchor_date: pd.Timestamp,
    degrees: Sequence[float] = tuple(TIME_DEGREES),
    year_days: float = SOLAR_YEAR_DAYS,
    cycles: int = 1,
) -> pd.DataFrame:
    """
    Gann time counts forward from a pivot, expressed as degrees of a
    solar year: 90 degrees is a quarter year, 180 half a year, 360 a full
    anniversary.
    """
    a = pd.Timestamp(anchor_date).normalize()
    rows = []
    for c in range(int(cycles)):
        for deg in degrees:
            total = c * 360.0 + float(deg)
            rows.append({
                "anchor": a,
                "degrees": total,
                "date": a + pd.Timedelta(float(total / 360.0 * year_days), "D"),
            })
    return pd.DataFrame(rows)


def vertical_markers(
    dates: Iterable,
    colour: str = "rgba(170,170,180,0.30)",
    width: int = 1,
    dash: str = "dot",
) -> List[Dict]:
    """Full-height vertical lines. yref='paper' avoids rescaling the y axis."""
    shapes = []
    for d in dates:
        if pd.isna(d):
            continue
        shapes.append({
            "type": "line", "xref": "x", "yref": "paper",
            "x0": pd.Timestamp(d), "y0": 0.0,
            "x1": pd.Timestamp(d), "y1": 1.0,
            "line": {"color": colour, "width": width, "dash": dash},
            "layer": "below",
        })
    return shapes


# --------------------------------------------------------------------------
# Planetary longitude to price
# --------------------------------------------------------------------------
def planet_price_series(
    lons: pd.DataFrame,
    body: str,
    mode: str = "scale",
    deg_to_price: float = 1.0,
    base_price: float = 100.0,
    harmonic: int = 0,
    offset: float = 0.0,
) -> pd.Series:
    """
    Convert a planet's longitude into a price line.

    mode='scale'
        price = longitude * deg_to_price + harmonic * 360 * deg_to_price + offset
        The straightforward linear conversion. `harmonic` shifts the line
        up or down by whole revolutions so it can be brought into the
        range of the chart.

    mode='sq9'
        price = (sqrt(base_price) + 2 * (longitude + harmonic*360) / 360) ** 2
        Maps longitude onto the Square of Nine spiral instead, so
        planetary degrees and Gann price levels share one geometry. This
        is the conversion Gann practitioners usually mean by "planets on
        price".

    Both produce sawtooth lines: longitude wraps at 360 degrees, so the
    line resets. That is expected, not a bug.
    """
    if body not in lons.columns:
        return pd.Series(dtype=float)

    lon = lons[body].to_numpy(dtype=float)
    h = float(harmonic)

    if mode == "sq9":
        base = max(float(base_price), 1e-6)
        root = math.sqrt(base) + 2.0 * (lon + h * 360.0) / 360.0
        vals = np.where(root > 0, root * root, np.nan) + float(offset)
    else:
        vals = (lon + h * 360.0) * float(deg_to_price) + float(offset)

    return pd.Series(vals, index=lons.index, name=f"{body}_price")


def fit_offset_to_price(planet_line: pd.Series, close: pd.Series) -> float:
    """
    Vertical shift that centres a planetary price line on the price series,
    so the two are visually comparable without hand-tuning an offset.
    """
    joined = pd.concat([planet_line.rename("p"), close.rename("c")], axis=1).dropna()
    if joined.empty:
        return 0.0
    return float(joined["c"].mean() - joined["p"].mean())


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------
def build_overlay(
    close: pd.Series,
    *,
    pivots: Optional[pd.DataFrame] = None,
    alignment_dates: Optional[Iterable] = None,
    ratios: Sequence[float] = (1 / 4, 1 / 2, 1.0, 2.0, 4.0),
    unit_price_per_day: Optional[float] = None,
    extend_days: int = 180,
    max_fan_anchors: int = 6,
    both_directions: bool = True,
    sq9_base: Optional[float] = None,
    sq9_degrees: Sequence[float] = (45, 90, 135, 180, 225, 270, 315, 360),
    sq9_rotations: int = 2,
    round_step: float = 0.0,
    show_time_cycles: bool = False,
    time_cycle_degrees: Sequence[float] = (90, 180, 270, 360),
    show_alignment_verticals: bool = True,
    max_verticals: int = 150,
    anchor_mode: str = "both",
    alignment_stride: int = 1,
    y_bounds: Optional[Tuple[float, float]] = None,
    y_pad_frac: float = 0.06,
    max_days: Optional[int] = None,
) -> Tuple[List[Dict], Dict[str, object]]:
    """
    Assemble every Gann shape for the chart in one pass.

    Returns (shapes, meta) where meta reports the choices actually made,
    so the GUI can display the 1x1 unit and anchor list instead of leaving
    them implicit.
    """
    s = close.dropna()
    if s.empty:
        return [], {}

    x0 = s.index.min()
    x_end = s.index.max() + pd.Timedelta(int(extend_days), "D")
    lo, hi = float(s.min()), float(s.max())

    unit = float(unit_price_per_day) if unit_price_per_day else auto_unit_price_per_day(s)

    # The band fans are allowed to occupy. Everything is clipped to it so the
    # price axis stays scaled to the price.
    if y_bounds is None:
        pad = (hi - lo) * float(y_pad_frac) if hi > lo else abs(hi) * 0.05 + 1.0
        y_bounds = (lo - pad, hi + pad)
    if max_days is None:
        max_days = fan_horizon_days(s, unit)

    shapes: List[Dict] = []
    meta: Dict[str, object] = {
        "unit_price_per_day": unit, "fan_anchors": [], "sq9_base": None,
        "y_bounds": (round(float(y_bounds[0]), 4), round(float(y_bounds[1]), 4)),
        "fan_horizon_days": int(max_days),
    }

    # Round-number horizontal grid
    if round_step and round_step > 0:
        levels = [v for v in round_number_levels(lo, hi, float(round_step))
                  if y_bounds[0] <= v <= y_bounds[1]]
        shapes.extend(horizontal_levels(levels, x0, x_end))
        meta["round_levels"] = len(levels)

    # Square of nine levels
    if sq9_base:
        sq9 = square_of_nine_levels(float(sq9_base), sq9_degrees, sq9_rotations)
        if not sq9.empty:
            # Filtered to the same band the fans are clipped to, so every
            # element of the overlay lives inside one agreed price window.
            inband = sq9[(sq9["price"] >= y_bounds[0]) & (sq9["price"] <= y_bounds[1])]
            shapes.extend(horizontal_levels(
                inband["price"].tolist(), x0, x_end,
                colour="rgba(220,170,60,0.30)", dash="dash",
            ))
            meta["sq9_base"] = float(sq9_base)
            meta["sq9_levels_in_view"] = int(len(inband))

    # Fans from price pivots
    anchors: List[Tuple[pd.Timestamp, float, str]] = []
    mode = str(anchor_mode).lower()
    if mode in ("pivots", "both") and pivots is not None and not pivots.empty:
        pv = pivots.copy()
        pv["extremity"] = np.where(
            pv["kind"] == "high",
            pv["price"] / max(hi, 1e-9),
            1.0 - pv["price"] / max(hi, 1e-9),
        )
        pv = pv.sort_values("extremity", ascending=False).head(int(max_fan_anchors))
        for _, r in pv.iterrows():
            anchors.append((pd.Timestamp(r["date"]), float(r["price"]), f"pivot-{r['kind']}"))

    # Fans from alignment dates, priced at the close of that day.
    #
    # `alignment_stride` takes every Nth alignment rather than thinning to
    # fill leftover slots, so the spacing between fans is something you set
    # rather than a side effect of how many pivots were found.
    if mode in ("alignments", "both") and alignment_dates is not None:
        al = pd.DatetimeIndex(pd.to_datetime(list(alignment_dates))).normalize()
        idx_norm = pd.DatetimeIndex(s.index).normalize()
        stride = max(int(alignment_stride), 1)
        room = max(int(max_fan_anchors) - len(anchors), 0)
        if room and len(al):
            for d in al[::stride][:room]:
                pos = int(np.searchsorted(idx_norm.values, np.datetime64(d), side="left"))
                if 0 <= pos < len(s):
                    anchors.append((s.index[pos], float(s.iloc[pos]), "alignment"))

    for a_date, a_price, kind in anchors:
        colour = ("rgba(120,200,255,0.42)" if kind == "alignment"
                  else "rgba(150,160,180,0.42)")
        ray_end = min(pd.Timestamp(x_end),
                      pd.Timestamp(a_date) + pd.Timedelta(int(max_days), "D"))
        shapes.extend(fan_lines(a_date, a_price, unit, ratios, ray_end,
                                both_directions, colour=colour,
                                y_bounds=y_bounds))
        meta["fan_anchors"].append({
            "date": str(pd.Timestamp(a_date).date()),
            "price": round(a_price, 4),
            "kind": kind,
        })

    # Vertical alignment markers. Capped: a fast pair with a wide orb can
    # produce hundreds of events, and every one becomes a shape. Past a
    # few hundred shapes Plotly gets sluggish and the chart is unreadable
    # anyway, so thin them evenly rather than dropping the tail.
    if show_alignment_verticals and alignment_dates is not None:
        al = list(alignment_dates)
        cap = max(int(max_verticals), 0)
        if cap and len(al) > cap:
            stride = int(np.ceil(len(al) / cap))
            al = al[::stride]
        shapes.extend(vertical_markers(al))
        meta["verticals_drawn"] = len(al)

    # Gann time counts from the strongest pivot
    if show_time_cycles and anchors:
        a_date = anchors[0][0]
        tc = time_cycle_dates(a_date, time_cycle_degrees, cycles=1)
        tc = tc[tc["date"] <= x_end]
        shapes.extend(vertical_markers(tc["date"].tolist(),
                                       colour="rgba(230,150,90,0.35)", dash="dashdot"))
        meta["time_cycle_anchor"] = str(pd.Timestamp(a_date).date())
        meta["time_cycle_dates"] = [str(d.date()) for d in tc["date"]]

    return shapes, meta


# --------------------------------------------------------------------------
# Angle-relative market state, the "rule of all angles", clusters
# --------------------------------------------------------------------------
# The functions below implement four specific claims from the Gann angle
# literature rather than general geometry. Each is stated as a rule that can
# be checked against data, which is the only reason they are worth coding:
#
#   1. Market strength can be read from which angle price is trading above.
#      Above the 1x1 is balanced, above the 2x1 is a strong uptrend, near the
#      1x2 is weaker, and anything under the 1x1 is weak.
#   2. The "rule of all angles": when price breaks one angle it travels to
#      the next one.
#   3. Where many angles from different anchors converge on a similar price,
#      that zone matters more. These are price clusters.
#   4. An angle crossing a retracement level marks a stronger point than
#      either alone.
#
# Rule 2 in particular is falsifiable, so `rule_of_all_angles` returns a hit
# rate rather than just drawing the lines. A rule that draws well and hits
# 50% of the time is a coin flip with good graphics.

def fan_line_values(
    dates: pd.DatetimeIndex,
    anchor_date: pd.Timestamp,
    anchor_price: float,
    unit_price_per_day: float,
    ratios: Optional[Dict[str, float]] = None,
    direction: int = 1,
    max_days: Optional[int] = None,
) -> pd.DataFrame:
    """
    Value of each fan line on each date, as numbers rather than shapes.

    Needed because everything below reasons about where price sits relative
    to the angles, which cannot be done with Plotly shape dictionaries.
    `direction` is +1 for a fan rising from a low, -1 for one falling from a
    high.

    `max_days` masks the projection once the fan has outrun the chart. Use
    `fan_horizon_days` to derive it. Without it a fan anchored five years
    back reports price as below every ratio, which reads as an extremely
    weak market when it actually means the fan expired.
    """
    ratios = ratios or FAN_RATIOS
    a_date = pd.Timestamp(anchor_date)
    days = np.asarray([(pd.Timestamp(d) - a_date).days for d in dates], dtype=float)
    out = {}
    for name, r in ratios.items():
        out[name] = float(anchor_price) + direction * float(unit_price_per_day) * float(r) * days
    frame = pd.DataFrame(out, index=pd.DatetimeIndex(dates))
    # Before the anchor, and past the useful horizon, the projection means
    # nothing.
    invalid = days < 0
    if max_days is not None:
        invalid = invalid | (days > float(max_days))
    frame[invalid] = np.nan
    return frame


def angle_state(
    close: pd.Series,
    anchor_date: pd.Timestamp,
    anchor_price: float,
    unit_price_per_day: float,
    ratios: Optional[Dict[str, float]] = None,
    direction: int = 1,
    max_days: Optional[int] = None,
) -> pd.DataFrame:
    """
    Where price sits within the fan, per bar.

    Returns `above` (how many fan lines price is above), `strength` (that
    count scaled to 0..1), `band` (the two lines price sits between) and
    `dist_1x1` (distance from the 1x1 in price units).

    `strength` is an ordinal count rather than a hand-assigned label per
    angle. The literature describes the reading qualitatively - balanced at
    the 1x1, strong above the 2x1, weak below - and counting lines crossed
    reproduces that ordering while giving something monotone that can be fed
    to a correlation or used as a calibration signal. Labels cannot be
    correlated with anything.
    """
    ratios = ratios or FAN_RATIOS
    c = close.dropna()
    vals = fan_line_values(c.index, anchor_date, anchor_price,
                           unit_price_per_day, ratios, direction, max_days)

    ordered = sorted(ratios.items(), key=lambda kv: kv[1])
    names = [n for n, _ in ordered]
    arr = vals[names].to_numpy(dtype=float)
    price = c.to_numpy(dtype=float).reshape(-1, 1)

    above = np.nansum(price > arr, axis=1).astype(float)
    valid = np.isfinite(arr).any(axis=1)
    above[~valid] = np.nan
    n_lines = len(names)

    bands = []
    for i in range(len(c)):
        if not valid[i]:
            bands.append(None)
            continue
        k = int(above[i])
        if k == 0:
            bands.append(f"below {names[0]}")
        elif k >= n_lines:
            bands.append(f"above {names[-1]}")
        else:
            bands.append(f"{names[k - 1]} to {names[k]}")

    one = vals["1x1"] if "1x1" in vals.columns else pd.Series(np.nan, index=c.index)
    return pd.DataFrame({
        "close": c,
        "above": above,
        "strength": above / max(n_lines, 1),
        "band": bands,
        "dist_1x1": c - one,
    }, index=c.index)


def rule_of_all_angles(
    close: pd.Series,
    anchor_date: pd.Timestamp,
    anchor_price: float,
    unit_price_per_day: float,
    ratios: Optional[Dict[str, float]] = None,
    direction: int = 1,
    max_bars_to_target: int = 120,
    max_days: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Test the claim that price travels to the next angle after breaking one.

    For every crossing of a fan line, the adjacent line in the direction of
    travel becomes the target. Records whether price reached it within
    `max_bars_to_target` bars and how long it took.

    Returns (events, summary) where summary carries the hit rate and the
    median bars to target.

    Two caveats the hit rate does not remove. Crossings are counted at the
    close, so an intrabar touch is missed and a wick through a line and back
    is not counted. And the hit rate has no null to compare against on its
    own: in a trending market price drifts toward the next line above simply
    because it drifts upward. Compare the figure against the same test on
    the offline demo random walk before reading anything into it.
    """
    ratios = ratios or FAN_RATIOS
    c = close.dropna()
    if len(c) < 20:
        return pd.DataFrame(), {"n_events": 0, "hit_rate": np.nan,
                                "control_hit_rate": np.nan, "edge": np.nan,
                                "median_bars": np.nan}

    vals = fan_line_values(c.index, anchor_date, anchor_price,
                           unit_price_per_day, ratios, direction, max_days)
    ordered = sorted(ratios.items(), key=lambda kv: kv[1])
    names = [n for n, _ in ordered]

    price = c.to_numpy(dtype=float)
    rows = []

    for j, name in enumerate(names):
        line = vals[name].to_numpy(dtype=float)
        rel = np.sign(price - line)
        for i in range(1, len(c)):
            if not (np.isfinite(rel[i]) and np.isfinite(rel[i - 1])):
                continue
            if rel[i] == rel[i - 1] or rel[i] == 0:
                continue

            up = rel[i] > 0
            nxt = j + 1 if up else j - 1
            if nxt < 0 or nxt >= len(names):
                continue
            target_name = names[nxt]
            target_line = vals[target_name].to_numpy(dtype=float)
            if not np.isfinite(target_line[i]):
                continue

            # Skip breaks where the target is already behind price. Near the
            # anchor every ratio converges, so a single bar can cross several
            # lines at once and the target is satisfied on the break bar
            # itself. Counting those inflated the hit rate to 0.64 with a
            # median of ZERO bars to target, which measured nothing but the
            # geometry of the fan's own vertex.
            if (up and price[i] >= target_line[i]) or \
               (not up and price[i] <= target_line[i]):
                continue

            end = min(i + int(max_bars_to_target), len(c) - 1)
            hit, bars = False, np.nan
            for k in range(i + 1, end + 1):
                if not np.isfinite(target_line[k]):
                    continue
                if (up and price[k] >= target_line[k]) or \
                   (not up and price[k] <= target_line[k]):
                    hit, bars = True, k - i
                    break

            rows.append({
                "date": c.index[i],
                "broke": name,
                "direction": "up" if up else "down",
                "price": round(float(price[i]), 4),
                "target_line": target_name,
                "target_price": round(float(target_line[i]), 4)
                if np.isfinite(target_line[i]) else np.nan,
                "reached": hit,
                "bars_to_target": bars,
            })

    events = pd.DataFrame(rows)
    if events.empty:
        return events, {"n_events": 0, "hit_rate": np.nan,
                        "control_hit_rate": np.nan, "edge": np.nan,
                        "median_bars": np.nan}

    events = events.sort_values("date").reset_index(drop=True)

    # Matched control: same break dates, same initial distance, but a static
    # target instead of a fan line. Without this the hit rate is unreadable,
    # because in any drifting market price tends to reach the next level up
    # simply by drifting.
    #
    # The comparison is informative but not perfectly matched, and the reason
    # is itself worth knowing: a fan line MOVES. An upward target recedes as
    # time passes, so chasing a rising 1x1 is strictly harder than reaching a
    # fixed level the same distance away, while a downward target descends to
    # meet price and is easier. Expect the rule to trail the control on up
    # breaks for that reason alone, before any question of predictive value.
    ctrl_hits = []
    for _, r in events.iterrows():
        if not np.isfinite(r["target_price"]):
            continue
        i = int(c.index.get_loc(r["date"]))
        dist = abs(float(r["target_price"]) - float(r["price"]))
        up = r["direction"] == "up"
        static_target = float(r["price"]) + (dist if up else -dist)
        seg = price[i + 1:min(i + 1 + int(max_bars_to_target), len(price))]
        if len(seg) == 0:
            continue
        ctrl_hits.append(bool((seg >= static_target).any() if up
                              else (seg <= static_target).any()))

    control = float(np.mean(ctrl_hits)) if ctrl_hits else np.nan
    hit = float(events["reached"].mean())

    summary = {
        "n_events": int(len(events)),
        "hit_rate": round(hit, 4),
        "control_hit_rate": round(control, 4) if np.isfinite(control) else np.nan,
        "edge": round(hit - control, 4) if np.isfinite(control) else np.nan,
        "median_bars": float(np.nanmedian(events["bars_to_target"].to_numpy(dtype=float))),
        "max_bars_allowed": int(max_bars_to_target),
    }
    return events, summary


def price_clusters(
    anchors: Sequence[Tuple[pd.Timestamp, float, int]],
    at_date: pd.Timestamp,
    unit_price_per_day: float,
    ratios: Optional[Dict[str, float]] = None,
    bin_width: Optional[float] = None,
    min_count: int = 3,
    price_window: Optional[Tuple[float, float]] = None,
    max_days: Optional[int] = None,
) -> pd.DataFrame:
    """
    Prices where fan lines from several anchors converge on one date.

    `anchors` is a sequence of (date, price, direction). Every line from
    every anchor is evaluated at `at_date`, binned, and bins holding at
    least `min_count` lines are returned as clusters, strongest first.

    Worth being clear about what raises a count here: adding more anchors or
    more ratios produces more clusters mechanically, not because the market
    changed. The count is only comparable between zones computed from the
    same anchor and ratio set.
    """
    ratios = ratios or FAN_RATIOS
    at = pd.Timestamp(at_date)
    idx = pd.DatetimeIndex([at])

    values = []
    for a_date, a_price, direction in anchors:
        if pd.Timestamp(a_date) > at:
            continue
        row = fan_line_values(idx, a_date, a_price, unit_price_per_day,
                              ratios, int(direction), max_days)
        for name in row.columns:
            v = float(row[name].iloc[0])
            if not np.isfinite(v):
                continue
            # Discard lines that have left the chart. Without a window the
            # bins span every projection out to plus and minus 1600 on an
            # instrument trading near 80, and one meaningless bin swallows
            # 184 lines.
            if price_window is not None and not (price_window[0] <= v <= price_window[1]):
                continue
            values.append({"price": v, "line": name,
                           "anchor": str(pd.Timestamp(a_date).date())})

    if not values:
        return pd.DataFrame(columns=["price_low", "price_high", "centre",
                                     "count", "lines"])

    frame = pd.DataFrame(values)
    if bin_width is None or bin_width <= 0:
        if price_window is not None:
            spread = float(price_window[1] - price_window[0])
        else:
            spread = float(frame["price"].max() - frame["price"].min())
        bin_width = max(spread / 60.0, 1e-9)

    frame["bin"] = np.floor(frame["price"] / bin_width).astype(int)
    out = []
    for b, grp in frame.groupby("bin"):
        if len(grp) < int(min_count):
            continue
        out.append({
            "price_low": round(float(grp["price"].min()), 4),
            "price_high": round(float(grp["price"].max()), 4),
            "centre": round(float(grp["price"].mean()), 4),
            "count": int(len(grp)),
            "lines": ", ".join(sorted(set(grp["line"]))),
        })
    if not out:
        return pd.DataFrame(columns=["price_low", "price_high", "centre",
                                     "count", "lines"])
    return (pd.DataFrame(out).sort_values("count", ascending=False)
            .reset_index(drop=True))


# Gann divided a range into eighths and thirds rather than using the
# Fibonacci ratios most platforms default to. The half is common to both and
# is the level the literature singles out for combining with an angle.
GANN_FRACTIONS: Dict[str, float] = {
    "1/8": 0.125, "1/4": 0.25, "1/3": 1 / 3, "3/8": 0.375,
    "1/2": 0.5,
    "5/8": 0.625, "2/3": 2 / 3, "3/4": 0.75, "7/8": 0.875,
}


def retracement_levels(
    high: float, low: float,
    fractions: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Retracement prices for a swing, in eighths and thirds."""
    fractions = fractions or GANN_FRACTIONS
    hi, lo = float(max(high, low)), float(min(high, low))
    rng = hi - lo
    rows = [{"name": n, "fraction": f, "price": round(lo + rng * f, 4)}
            for n, f in fractions.items()]
    return pd.DataFrame(rows).sort_values("price").reset_index(drop=True)


def angle_retracement_confluence(
    dates: pd.DatetimeIndex,
    anchor_date: pd.Timestamp,
    anchor_price: float,
    unit_price_per_day: float,
    retracements: pd.DataFrame,
    ratios: Optional[Dict[str, float]] = None,
    direction: int = 1,
    tolerance: Optional[float] = None,
) -> pd.DataFrame:
    """
    Dates and prices where a fan line crosses a retracement level.

    These are the intersection points the literature calls out as stronger
    than either component. Purely geometric: both inputs are known in
    advance, so the crossings are computable into the future.
    """
    ratios = ratios or FAN_RATIOS
    vals = fan_line_values(dates, anchor_date, anchor_price,
                           unit_price_per_day, ratios, direction)
    if retracements.empty:
        return pd.DataFrame(columns=["date", "line", "level", "price"])

    if tolerance is None:
        tolerance = abs(float(unit_price_per_day)) * 0.75 + 1e-9

    rows = []
    for _, lvl in retracements.iterrows():
        target = float(lvl["price"])
        for name in vals.columns:
            series = vals[name].to_numpy(dtype=float)
            diff = series - target
            for i in range(1, len(diff)):
                if not (np.isfinite(diff[i]) and np.isfinite(diff[i - 1])):
                    continue
                crossed = np.sign(diff[i]) != np.sign(diff[i - 1])
                if crossed or abs(diff[i]) <= tolerance:
                    rows.append({
                        "date": pd.DatetimeIndex(dates)[i],
                        "line": name,
                        "level": lvl["name"],
                        "price": round(target, 4),
                    })
                    break
    if not rows:
        return pd.DataFrame(columns=["date", "line", "level", "price"])
    return (pd.DataFrame(rows).drop_duplicates(subset=["line", "level"])
            .sort_values("date").reset_index(drop=True))


def square_the_range(
    anchor_date: pd.Timestamp,
    high: float,
    low: float,
    unit_price_per_day: float,
    multiples: Sequence[float] = (0.25, 0.5, 1.0, 1.5, 2.0),
) -> pd.DataFrame:
    """
    Dates where elapsed time equals the swing range, in matching units.

    Gann's squaring: convert the price range into time units by dividing by
    the 1x1 rate, then project that many days forward from the anchor. A
    turn is expected where time and price balance.

    The literature notes this reads better on weekly and monthly data,
    because a daily chart offers so many candidate tops, bottoms and ranges
    that some projection will land near a turn by coincidence. Use the
    interval selector rather than fighting that on daily bars.
    """
    unit = abs(float(unit_price_per_day))
    rng = abs(float(high) - float(low))
    if unit <= 0 or rng <= 0:
        return pd.DataFrame(columns=["multiple", "days", "date"])
    base_days = rng / unit
    a = pd.Timestamp(anchor_date).normalize()
    rows = [{"multiple": float(m),
             "days": int(round(base_days * float(m))),
             "date": a + pd.Timedelta(int(round(base_days * float(m))), "D")}
            for m in multiples]
    return pd.DataFrame(rows)


def harmonise_unit(unit: float, divisors_of_360: bool = True) -> float:
    """
    Snap the 1x1 rate to a value that divides the circle evenly.

    Practitioners fit a rough rate from the chart and then round it to a
    harmonic number - preferring 60 over 59 or 61 because 60 divides 360 -
    so that the price scale and the degree scale share a common measure.
    Scales by powers of ten first, so it works whether the instrument
    trades near 1 or near 7000.
    """
    u = abs(float(unit))
    if u <= 0:
        return 1.0
    if not divisors_of_360:
        return u
    harmonics = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24, 30,
                 36, 40, 45, 60, 72, 90, 120, 180, 360]
    scale = 10.0 ** np.floor(np.log10(u))
    normalised = u / scale
    best = min(harmonics, key=lambda h: abs(h - normalised * 10) )
    candidate = best / 10.0 * scale
    # Never move the unit by more than a factor of two, or the snap stops
    # being a rounding and becomes a different rate entirely.
    if candidate <= 0 or not (0.5 <= candidate / u <= 2.0):
        return u
    return float(candidate)

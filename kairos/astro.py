"""
kairos.astro
============
Planetary geometry engine.

Why `ephem` and not `skyfield`
------------------------------
The original build used Skyfield, which requires the DE421 SPK kernel
(~17 MB) to be downloaded from NASA on first run. That is fatal for a
frozen one-file .exe: the download often fails behind a firewall, the
file lands in a temp directory that is wiped on exit, and PyInstaller
cannot see it at build time.

`ephem` (PyEphem) embeds VSOP87 for the planets and the Chapront lunar
theory in the compiled extension. No data files, no network, ~1 MB, and
it resolves ~10 bodies x 6500 days in under 2 seconds. Accuracy is a few
arc-seconds, which is three orders of magnitude finer than the 0.5-5
degree orbs any aspect study uses.

Everything here is pure geometry. It is deterministic and computable
arbitrarily far into the future, which is what makes the forward
projection in the GUI possible.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import ephem
import numpy as np
import pandas as pd

DEG = 180.0 / math.pi

# --------------------------------------------------------------------------
# Bodies
# --------------------------------------------------------------------------
BODIES: List[str] = [
    "SUN", "MOON", "MERCURY", "VENUS", "MARS",
    "JUPITER", "SATURN", "URANUS", "NEPTUNE", "PLUTO",
]

# Heliocentric frame swaps the Sun for the Earth and drops the Moon,
# because "heliocentric longitude of the Sun" is meaningless.
HELIO_BODIES: List[str] = [
    "EARTH", "MERCURY", "VENUS", "MARS",
    "JUPITER", "SATURN", "URANUS", "NEPTUNE", "PLUTO",
]

_EPHEM_CLASS = {
    "SUN": ephem.Sun,
    "MOON": ephem.Moon,
    "MERCURY": ephem.Mercury,
    "VENUS": ephem.Venus,
    "MARS": ephem.Mars,
    "JUPITER": ephem.Jupiter,
    "SATURN": ephem.Saturn,
    "URANUS": ephem.Uranus,
    "NEPTUNE": ephem.Neptune,
    "PLUTO": ephem.Pluto,
}

# Slow bodies carry more weight: a Jupiter-Saturn square sits in orb for
# months, a Moon-Mercury square for a couple of hours.
BODY_WEIGHT: Dict[str, float] = {
    "MOON": 0.25, "MERCURY": 0.50, "SUN": 0.80, "VENUS": 0.70,
    "MARS": 0.85, "JUPITER": 1.00, "SATURN": 1.00, "URANUS": 0.90,
    "NEPTUNE": 0.85, "PLUTO": 0.85, "EARTH": 0.80,
}

# Classical + Gann-favoured harmonics of the circle.
ASPECTS_DEFAULT: List[float] = [0.0, 45.0, 60.0, 90.0, 120.0, 135.0, 144.0, 180.0]
ASPECTS_ALL: List[float] = [
    0.0, 30.0, 36.0, 45.0, 51.4286, 60.0, 72.0, 90.0,
    102.857, 108.0, 120.0, 135.0, 144.0, 150.0, 180.0,
]

ASPECT_WEIGHT: Dict[float, float] = {
    0.0: 1.00, 180.0: 1.00, 90.0: 0.85, 120.0: 0.80, 60.0: 0.60,
    45.0: 0.45, 135.0: 0.45, 144.0: 0.50, 72.0: 0.40, 36.0: 0.35,
    30.0: 0.30, 150.0: 0.30, 108.0: 0.25, 51.4286: 0.20, 102.857: 0.20,
}

ASPECT_NAME: Dict[float, str] = {
    0.0: "conjunction", 30.0: "semisextile", 36.0: "decile", 45.0: "semisquare",
    51.4286: "septile", 60.0: "sextile", 72.0: "quintile", 90.0: "square",
    102.857: "biseptile", 108.0: "tredecile", 120.0: "trine",
    135.0: "sesquiquadrate", 144.0: "biquintile", 150.0: "quincunx",
    180.0: "opposition",
}

SIGNS = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
         "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]


def aspect_label(angle: float) -> str:
    """Human name for an aspect angle, falling back to the raw degrees."""
    key = min(ASPECT_NAME, key=lambda a: abs(a - angle))
    if abs(key - angle) < 0.05:
        return ASPECT_NAME[key]
    return f"{angle:g}deg"


def aspect_weight(angle: float) -> float:
    key = min(ASPECT_WEIGHT, key=lambda a: abs(a - angle))
    if abs(key - angle) < 0.05:
        return ASPECT_WEIGHT[key]
    return 0.3


def sign_of(longitude: float) -> str:
    return SIGNS[int(longitude // 30) % 12]


def dms(longitude: float) -> str:
    """Zodiacal notation, e.g. 144.72 -> '24 Leo 43'."""
    lon = float(longitude) % 360.0
    within = lon % 30.0
    deg = int(within)
    minutes = int(round((within - deg) * 60))
    if minutes == 60:
        deg, minutes = deg + 1, 0
    return f"{deg:02d} {sign_of(lon)[:3]} {minutes:02d}"


# --------------------------------------------------------------------------
# Longitudes
# --------------------------------------------------------------------------
def _to_ephem_dates(index: pd.DatetimeIndex) -> List[ephem.Date]:
    """ephem.Date wants naive UTC datetimes; strip any tz first."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return [ephem.Date(ts.to_pydatetime()) for ts in idx]


def longitudes(
    dates: pd.DatetimeIndex,
    bodies: Optional[Sequence[str]] = None,
    frame: str = "geocentric",
) -> pd.DataFrame:
    """
    Ecliptic longitude in degrees [0, 360) for each body on each date.

    frame='geocentric'  -> apparent longitude as seen from Earth (what
                           astrologers and Gann used).
    frame='heliocentric'-> longitude as seen from the Sun (what the
                           Bradley siderograph uses). EARTH replaces SUN.
    """
    dates = pd.DatetimeIndex(dates)
    if len(dates) == 0:
        return pd.DataFrame(index=dates)

    helio = frame.lower().startswith("helio")
    if bodies is None:
        bodies = HELIO_BODIES if helio else BODIES
    bodies = [b.upper() for b in bodies]

    edates = _to_ephem_dates(dates)
    out: Dict[str, np.ndarray] = {b: np.empty(len(dates), dtype=float) for b in bodies}

    # Instantiate once, recompute per date. Cheap; the constructor is the
    # expensive part in ephem.
    inst = {}
    for b in bodies:
        if b == "EARTH":
            inst[b] = ephem.Sun()          # Earth helio lon = Sun geo lon + 180
        else:
            inst[b] = _EPHEM_CLASS[b]()

    for i, d in enumerate(edates):
        for b in bodies:
            body = inst[b]
            body.compute(d)
            if b == "EARTH":
                lon = math.degrees(float(ephem.Ecliptic(body).lon)) + 180.0
            elif helio and b not in ("SUN", "MOON"):
                lon = math.degrees(float(body.hlon))
            else:
                lon = math.degrees(float(ephem.Ecliptic(body).lon))
            out[b][i] = lon % 360.0

    return pd.DataFrame(out, index=dates)


def daily_speed(lons: pd.DataFrame) -> pd.DataFrame:
    """
    Longitude change per day, unwrapped across the 360 boundary.
    Negative values are retrograde motion.
    """
    days = np.asarray(
        (lons.index - lons.index[0]).total_seconds() / 86400.0, dtype=float
    )
    spd = {}
    for col in lons.columns:
        unwrapped = np.unwrap(np.deg2rad(lons[col].to_numpy(dtype=float))) * DEG
        spd[col] = np.gradient(unwrapped, days) if len(days) > 1 else np.zeros(len(days))
    return pd.DataFrame(spd, index=lons.index)


def separation(lon_a: np.ndarray, lon_b: np.ndarray) -> np.ndarray:
    """Absolute angular separation folded into [0, 180]."""
    a = np.asarray(lon_a, dtype=float).ravel()
    b = np.asarray(lon_b, dtype=float).ravel()
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def all_pairs(bodies: Sequence[str]) -> List[Tuple[str, str]]:
    bl = list(bodies)
    return [(bl[i], bl[j]) for i in range(len(bl)) for j in range(i + 1, len(bl))]


def pair_weight(a: str, b: str) -> float:
    return BODY_WEIGHT.get(a.upper(), 0.7) * BODY_WEIGHT.get(b.upper(), 0.7)


# --------------------------------------------------------------------------
# Discrete events
# --------------------------------------------------------------------------
@dataclass
class EventSpec:
    """Parameters controlling which alignments count as events."""
    aspects: Sequence[float] = tuple(ASPECTS_DEFAULT)
    orb_deg: float = 2.0
    min_separation_days: int = 3
    pairs: Optional[Sequence[Tuple[str, str]]] = None


def aspect_events(
    lons: pd.DataFrame,
    spec: Optional[EventSpec] = None,
) -> pd.DataFrame:
    """
    Dates on which a body pair is closest to an exact aspect.

    An event is a local minimum of |separation - aspect| that also sits
    inside the orb. Taking local minima rather than every in-orb day is
    what stops a slow Jupiter-Saturn square from producing 200 duplicate
    "events" and swamping every fast-moving pair in the study.

    Returns columns: date, a, b, aspect, aspect_name, pair, separation,
    offset (degrees from exact), weight.
    """
    spec = spec or EventSpec()
    pairs = list(spec.pairs) if spec.pairs else all_pairs(list(lons.columns))
    orb = float(spec.orb_deg)
    if orb <= 0:
        orb = 0.5

    dates = pd.DatetimeIndex(lons.index)
    day_num = np.asarray((dates - dates[0]).days, dtype=float)
    rows = []

    for a, b in pairs:
        if a not in lons.columns or b not in lons.columns:
            continue
        sep = separation(lons[a].to_numpy(), lons[b].to_numpy())
        pw = pair_weight(a, b)
        for asp in spec.aspects:
            d = np.abs(sep - float(asp))
            if d.size < 3:
                continue
            core = d[1:-1]
            is_min = (core <= d[:-2]) & (core <= d[2:]) & (core <= orb)
            idxs = np.flatnonzero(is_min) + 1
            last_day = -1e18
            for i in idxs:
                if day_num[i] - last_day < spec.min_separation_days:
                    continue
                last_day = day_num[i]
                exactness = 1.0 - (d[i] / orb)
                rows.append({
                    "date": dates[i],
                    "a": a,
                    "b": b,
                    "pair": f"{a}-{b}",
                    "aspect": float(asp),
                    "aspect_name": aspect_label(float(asp)),
                    "separation": float(sep[i]),
                    "offset": float(d[i]),
                    "weight": float(pw * aspect_weight(float(asp)) * max(exactness, 0.05)),
                })

    if not rows:
        return pd.DataFrame(columns=[
            "date", "a", "b", "pair", "aspect", "aspect_name",
            "separation", "offset", "weight",
        ])
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def station_events(lons: pd.DataFrame, skip: Sequence[str] = ("SUN", "MOON", "EARTH")) -> pd.DataFrame:
    """
    Retrograde and direct stations: dates where daily motion changes sign.
    Gann treated these as turning-point candidates in their own right.
    """
    spd = daily_speed(lons)
    dates = pd.DatetimeIndex(lons.index)
    rows = []
    for col in lons.columns:
        if col.upper() in {s.upper() for s in skip}:
            continue
        v = spd[col].to_numpy(dtype=float)
        flip = np.flatnonzero(np.sign(v[:-1]) * np.sign(v[1:]) < 0) + 1
        for i in flip:
            rows.append({
                "date": dates[i],
                "body": col,
                "kind": "retrograde" if v[i] < 0 else "direct",
                "longitude": float(lons[col].iloc[i]),
                "weight": BODY_WEIGHT.get(col.upper(), 0.7),
            })
    if not rows:
        return pd.DataFrame(columns=["date", "body", "kind", "longitude", "weight"])
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def ingress_events(lons: pd.DataFrame, arc_deg: float = 30.0) -> pd.DataFrame:
    """Dates a body crosses a 30-degree (sign) boundary, or any chosen arc."""
    dates = pd.DatetimeIndex(lons.index)
    rows = []
    for col in lons.columns:
        v = lons[col].to_numpy(dtype=float)
        bucket = np.floor(v / float(arc_deg)).astype(int)
        change = np.flatnonzero(bucket[1:] != bucket[:-1]) + 1
        for i in change:
            rows.append({
                "date": dates[i],
                "body": col,
                "kind": "ingress",
                "longitude": float(v[i]),
                "sign": sign_of(v[i]),
                "weight": BODY_WEIGHT.get(col.upper(), 0.7) * 0.5,
            })
    if not rows:
        return pd.DataFrame(columns=["date", "body", "kind", "longitude", "sign", "weight"])
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------
# Continuous indices
# --------------------------------------------------------------------------
def harmonic_index(
    lons: pd.DataFrame,
    pairs: Optional[Sequence[Tuple[str, str]]] = None,
    harmonic: int = 1,
    weighted: bool = True,
    normalise: bool = True,
) -> pd.Series:
    """
    Smooth continuous alignment wave: sum of cos(h * separation) over pairs.

    This is the Bradley siderograph idea. Unlike discrete events there is
    no orb and no threshold, so nothing is thrown away: the signal rises
    and falls continuously as the geometry tightens and loosens. Harmonic
    1 peaks on conjunctions, 2 on conjunctions and oppositions, 4 adds
    the squares, and so on.
    """
    pairs = list(pairs) if pairs else all_pairs(list(lons.columns))
    total = np.zeros(len(lons), dtype=float)
    wsum = 0.0
    for a, b in pairs:
        if a not in lons.columns or b not in lons.columns:
            continue
        sep = separation(lons[a].to_numpy(), lons[b].to_numpy())
        w = pair_weight(a, b) if weighted else 1.0
        total += w * np.cos(np.deg2rad(harmonic * sep))
        wsum += w
    if normalise and wsum > 0:
        total /= wsum
    return pd.Series(total, index=lons.index, name=f"harmonic_{harmonic}")


def declination_index(dates: pd.DatetimeIndex, bodies: Sequence[str] = ("SUN", "VENUS", "MARS")) -> pd.Series:
    """
    Mean declination of the chosen bodies. Included because declination
    (north/south of the celestial equator) is a separate axis from
    longitude and some Gann practitioners weight it heavily.
    """
    dates = pd.DatetimeIndex(dates)
    edates = _to_ephem_dates(dates)
    acc = np.zeros(len(dates), dtype=float)
    n = 0
    for b in bodies:
        cls = _EPHEM_CLASS.get(b.upper())
        if cls is None:
            continue
        body = cls()
        vals = np.empty(len(dates), dtype=float)
        for i, d in enumerate(edates):
            body.compute(d)
            vals[i] = math.degrees(float(body.dec))
        acc += vals
        n += 1
    if n:
        acc /= n
    return pd.Series(acc, index=dates, name="declination")


_MEAN_MOTION_CACHE: Dict[str, Dict[str, float]] = {}


def mean_motions(frame: str = "heliocentric", span_years: int = 200) -> Dict[str, float]:
    """
    Mean longitude motion in degrees per day for every body, fitted over a
    long baseline so that retrograde loops and short-span noise average
    out. Sampled every 10 days, which is ample for a linear fit.

    Cached, because a 200-year fit is the one genuinely slow call here.
    """
    key = f"{frame}:{span_years}"
    if key in _MEAN_MOTION_CACHE:
        return _MEAN_MOTION_CACHE[key]

    start = pd.Timestamp("1950-01-01")
    idx = pd.date_range(start, start + pd.Timedelta(int(span_years * 365.2422), "D"), freq="10D")
    lons = longitudes(idx, frame=frame)
    days = np.asarray((idx - idx[0]).total_seconds() / 86400.0, dtype=float)

    out: Dict[str, float] = {}
    for col in lons.columns:
        unwrapped = np.unwrap(np.deg2rad(lons[col].to_numpy(dtype=float))) * DEG
        slope = np.polyfit(days, unwrapped, 1)[0]
        out[col] = float(slope)
    _MEAN_MOTION_CACHE[key] = out
    return out


def synodic_periods(
    pairs: Optional[Sequence[Tuple[str, str]]] = None,
    bodies: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Synodic (conjunction-to-conjunction) period for each pair, in days.

    Computed from long-baseline mean motions rather than from the loaded
    date range. Two reasons that matters:

      * A 3-year window cannot measure a Jupiter-Saturn cycle that takes
        20 years to close, and a naive fit returns nonsense.
      * Geocentric relative motion is the wrong frame. Sun and Mercury
        share a mean geocentric motion, so their relative longitude never
        accumulates and the geocentric "period" diverges. The real 116-day
        Mercury cycle only appears heliocentrically.

    So pairs are measured heliocentrically with EARTH standing in for SUN,
    except pairs involving the MOON, which are inherently geocentric.
    Sanity check against published values: Venus-Jupiter 237 d,
    Jupiter-Saturn 7254 d, Mercury-Earth 116 d, Moon-Sun 29.5 d.
    """
    if pairs is None:
        pairs = all_pairs(list(bodies) if bodies else BODIES)

    helio = mean_motions("heliocentric")
    geo = mean_motions("geocentric")

    def motion(body: str, use_geo: bool) -> Optional[float]:
        b = body.upper()
        if use_geo:
            return geo.get(b)
        if b == "SUN":
            return helio.get("EARTH")
        if b == "MOON":
            return None
        return helio.get(b)

    rows = []
    for a, b in pairs:
        use_geo = "MOON" in (a.upper(), b.upper())
        na, nb = motion(a, use_geo), motion(b, use_geo)
        if na is None or nb is None:
            continue
        rel = abs(na - nb)
        if rel < 1e-9:
            continue
        period = 360.0 / rel
        rows.append({
            "pair": f"{a}-{b}", "a": a, "b": b,
            "synodic_days": float(period),
            "synodic_years": float(period / 365.2422),
            "frame": "geocentric" if use_geo else "heliocentric",
        })
    if not rows:
        return pd.DataFrame(columns=["pair", "a", "b", "synodic_days", "synodic_years", "frame"])
    return pd.DataFrame(rows).sort_values("synodic_days").reset_index(drop=True)


# --------------------------------------------------------------------------
# Convenience
# --------------------------------------------------------------------------
def calendar_index(start, end) -> pd.DatetimeIndex:
    """Every calendar day in the range, normalised to midnight."""
    return pd.date_range(pd.Timestamp(start).normalize(),
                         pd.Timestamp(end).normalize(), freq="D")


def upcoming_events(
    start,
    days_ahead: int = 365,
    bodies: Optional[Sequence[str]] = None,
    spec: Optional[EventSpec] = None,
    frame: str = "geocentric",
) -> pd.DataFrame:
    """
    Forward-looking alignment calendar. The ephemeris is deterministic, so
    this is exact geometry rather than a forecast of anything.
    """
    start = pd.Timestamp(start).normalize()
    idx = calendar_index(start, start + pd.Timedelta(int(days_ahead), "D"))
    lons = longitudes(idx, bodies=bodies, frame=frame)
    ev = aspect_events(lons, spec)
    if not ev.empty:
        ev = ev[ev["date"] >= start].reset_index(drop=True)
    return ev

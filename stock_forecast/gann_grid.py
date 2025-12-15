# stock_forecast/gann_grid.py
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from stock_forecast.dataset import _planet_positions
PLANETS = ['SUN','MERCURY','VENUS','MARS','JUPITER','SATURN']
def _sep_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs((a - b + 180.0) % 360.0 - 180.0).astype(float)
def _nearest_aspect_distance(sep: np.ndarray, aspects: List[float]) -> np.ndarray:
    dmin = np.full_like(sep, 9999.0, dtype=float)
    for a in aspects:
        dmin = np.minimum(dmin, np.abs(sep - float(a)))
    return dmin
def alignment_indices(
    dates: pd.DatetimeIndex,
    pair: Tuple[str, str] = ("VENUS", "JUPITER"),
    aspects_deg: List[float] = [0,60,90,120,180],
    orb_deg: float = 2.0,
    min_separation_days: int = 5
) -> List[int]:
    longs = _planet_positions(dates)
    a, b = pair
    sep = _sep_deg(longs[a], longs[b])
    dmin = _nearest_aspect_distance(sep, aspects_deg)
    mask = dmin <= float(orb_deg)
    idxs = np.where(mask)[0].tolist()
    kept, last = [], -10**9
    for i in idxs:
        if 0 < i < len(dmin)-1 and dmin[i] <= dmin[i-1] and dmin[i] <= dmin[i+1]:
            if (i - last) >= int(min_separation_days):
                kept.append(i); last = i
    return kept
def round_price_levels(y: np.ndarray, step: float) -> List[float]:
    if step <= 0: return []
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    start = np.floor(ymin / step) * step
    out, v = [], start
    while v <= ymax:
        out.append(float(v)); v += step
    return out
def _base_slope_1x1(dates: pd.DatetimeIndex, y: np.ndarray) -> float:
    if len(dates) < 2: return 0.0
    days = max((dates[-1] - dates[0]).days, 1)
    pr = float(np.nanmax(y) - np.nanmin(y))
    return pr / days if days > 0 else 0.0
def build_fan_lines(
    dates: pd.DatetimeIndex,
    y: np.ndarray,
    anchor_idx: int,
    ratios: List[float],
    slope_scale: float = 1.0,
    extend_days: int = 0,
    both_dirs: bool = True
) -> List[Dict]:
    lines: List[Dict] = []
    y0 = float(y[anchor_idx]); x0 = dates[anchor_idx]
    base = _base_slope_1x1(dates, y) * float(slope_scale)
    end_date = dates[-1] + pd.Timedelta(days=int(extend_days))
    total_days = (end_date - x0).days
    if total_days <= 0 or base == 0.0:
        return lines
    for r in ratios:
        slope = base * float(r)
        for sgn in ([1,-1] if both_dirs else [1]):
            y1 = y0 + sgn * slope * total_days
            lines.append(dict(
                type="line", x0=x0, y0=y0, x1=end_date, y1=y1,
                line=dict(color="rgba(180,180,180,0.45)", width=1)
            ))
    return lines
def build_overlay_shapes(
    dates: pd.DatetimeIndex,
    close: pd.Series,
    pair: Tuple[str, str] = ("VENUS","JUPITER"),
    aspects_deg: List[float] = [0,60,90,120,180],
    orb_deg: float = 2.0,
    max_anchors: int = 36,
    ratios: List[float] = [1/8,1/4,1/3,1/2,1,2,3,4,8],
    slope_scale: float = 1.2,
    extend_days: int = 180,
    both_dirs: bool = True,
    add_verticals: bool = True,
    price_step: float = 72.0
) -> List[Dict]:
    y = close.values.astype(float)
    shapes: List[Dict] = []
    if price_step and price_step > 0:
        levels = round_price_levels(y, price_step)
        end_line = dates[-1] + pd.Timedelta(days=int(extend_days))
        for lv in levels:
            shapes.append(dict(
                type="line", x0=dates[0], y0=lv, x1=end_line, y1=lv,
                line=dict(color="rgba(60,100,200,0.20)", width=1)
            ))
    idxs = alignment_indices(dates, pair, aspects_deg, orb_deg, min_separation_days=5)
    if len(idxs) == 0: return shapes
    if max_anchors > 0 and len(idxs) > max_anchors:
        step = max(len(idxs)//max_anchors, 1); idxs = idxs[::step][:max_anchors]
    for ix in idxs:
        if add_verticals:
            shapes.append(dict(
                type="line", x0=dates[ix], y0=float(np.nanmin(y)), x1=dates[ix], y1=float(np.nanmax(y)),
                line=dict(color="rgba(160,160,160,0.25)", width=1, dash="dot")
            ))
        shapes.extend(build_fan_lines(dates, y, ix, ratios, slope_scale, extend_days, both_dirs))
    return shapes

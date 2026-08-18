"""
kairos.charting
===============
Plotly figure construction, kept out of the GUI so the figures can be
built and tested without a Streamlit session.

The main chart puts price on the left axis and the alignment wave on the
right. The original build rescaled the wave into the price range and drew
it on the same axis, which makes an arbitrary scaling choice look like a
meaningful overlay — the two lines appear to track each other because
they were forced onto the same range. A second axis keeps both readable
and keeps the comparison honest.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

PALETTE = ["#4FA8E0", "#E8973A", "#57B96C", "#A574CE", "#D9524B",
           "#3FB0A0", "#C9B037", "#7C8CF0", "#E06C9F", "#8FBF5A"]

GRID = "rgba(128,128,128,0.15)"


def _layout(fig: go.Figure, height: int = 720, title: Optional[str] = None) -> go.Figure:
    fig.update_layout(
        height=height,
        margin={"l": 8, "r": 8, "t": 40 if title else 16, "b": 8},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01,
                "xanchor": "left", "x": 0,
                # Pinned to an empty string rather than left unset. An unset
                # legend title serialises as an empty object, and a theme
                # layer that then rewrites the layout can leave a stray
                # "undefined" label drawn over the first legend swatch.
                "title": {"text": ""}},
        template="plotly_dark",
        # gridcolor only. Setting a bare yaxis dict here would be a
        # recursive merge, but listing range/autorange again would clobber
        # the pinned axis set by price_chart, so they are deliberately absent.
        xaxis={"gridcolor": GRID, "rangeslider": {"visible": False}},
        yaxis={"gridcolor": GRID},
    )
    # Only set a title when there is one. Passing title=None produces
    # layout.title = {}, an empty object rather than no title at all, which
    # some renderers display as a blank or literal "undefined" heading.
    if title:
        fig.update_layout(title={"text": title})
    return fig


def price_chart(
    df: pd.DataFrame,
    *,
    shapes: Optional[List[Dict]] = None,
    planet_lines: Optional[Dict[str, pd.Series]] = None,
    wave: Optional[pd.Series] = None,
    wave_name: str = "Alignment wave",
    event_markers: Optional[pd.DataFrame] = None,
    extend_days: int = 0,
    candles: bool = True,
    height: int = 760,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Price with Gann geometry, planetary price lines, and the alignment
    wave on a secondary axis.
    """
    fig = go.Figure()
    if df is None or df.empty:
        return _layout(fig, height, title or "No data")

    idx = df.index

    if candles and {"Open", "High", "Low"}.issubset(df.columns) and df["Open"].notna().any():
        # showlegend=False on purpose. A candlestick legend entry has no
        # useful label - the candles are self-evident, and the price scale is
        # named on the y axis instead - and the entry was the source of a
        # stray "undefined" string drawn over its own colour swatch. Removing
        # the entry removes the artifact outright rather than hoping a
        # renderer labels it correctly. increasing/decreasing are given as
        # explicit dicts for the same reason: the magic-underscore form
        # builds those sub-objects implicitly, and implicit is what got
        # rewritten by the theme layer.
        fig.add_trace(go.Candlestick(
            x=idx, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name="Price",
            showlegend=False,
            increasing={"line": {"color": "#3DBE86"}, "fillcolor": "#3DBE86"},
            decreasing={"line": {"color": "#E0524B"}, "fillcolor": "#E0524B"},
        ))
    else:
        fig.add_trace(go.Scatter(
            x=idx, y=df["Close"], mode="lines", name="Close", showlegend=True,
            line={"color": "#E8E8E8", "width": 1.6},
        ))

    if planet_lines:
        for i, (name, series) in enumerate(planet_lines.items()):
            s = series.dropna()
            if s.empty:
                continue
            fig.add_trace(go.Scatter(
                x=s.index, y=s.values, mode="lines", name=name,
                line={"color": PALETTE[i % len(PALETTE)], "width": 1},
                opacity=0.55, hovertemplate="%{y:.2f}<extra>" + name + "</extra>",
            ))

    if wave is not None:
        w = wave.dropna()
        if not w.empty:
            fig.add_trace(go.Scatter(
                x=w.index, y=w.values, mode="lines", name=wave_name,
                line={"color": "#F2C14E", "width": 1.8}, yaxis="y2", opacity=0.9,
            ))
            fig.update_layout(yaxis2={
                "title": wave_name, "overlaying": "y", "side": "right",
                "showgrid": False, "zeroline": True,
                "zerolinecolor": "rgba(242,193,78,0.35)",
            })

    if event_markers is not None and not event_markers.empty:
        close = df["Close"]
        idx_norm = pd.DatetimeIndex(close.index).normalize()
        ev = event_markers.copy()
        ev["date"] = pd.to_datetime(ev["date"]).dt.normalize()
        pos = np.searchsorted(idx_norm.values, ev["date"].values, side="left")
        keep = (pos >= 0) & (pos < len(close))
        if keep.any():
            fig.add_trace(go.Scatter(
                x=close.index[pos[keep]],
                y=close.iloc[pos[keep]].values,
                mode="markers", name="Alignments",
                marker={"size": 7, "symbol": "diamond", "color": "#7FD4FF",
                        "line": {"width": 1, "color": "#0C2A3A"}},
                text=ev.loc[keep, "label"].values if "label" in ev.columns else None,
                hovertemplate="%{text}<br>%{x|%Y-%m-%d}<extra></extra>",
            ))

    if shapes:
        fig.update_layout(shapes=shapes)

    x_end = idx.max() + pd.Timedelta(int(extend_days), "D") if extend_days else idx.max()

    # The price axis is pinned to the price data rather than left to
    # autorange. Plotly folds shape coordinates into its autorange, so a
    # single steep fan ray or a far Square of Nine level silently rescales
    # the axis and flattens the candles into a line. Fan rays are clipped at
    # the band in gann.fan_lines; this makes the axis itself immovable so
    # anything that escapes clipping is cropped rather than allowed to take
    # over the chart.
    y_lo, y_hi = float(df["Low"].min()), float(df["High"].max())
    if not np.isfinite(y_lo) or not np.isfinite(y_hi):
        y_lo, y_hi = float(df["Close"].min()), float(df["Close"].max())
    for series in (planet_lines or {}).values():
        s = series.dropna()
        if not s.empty:
            y_lo = min(y_lo, float(s.min()))
            y_hi = max(y_hi, float(s.max()))
    pad = (y_hi - y_lo) * 0.06 if y_hi > y_lo else abs(y_hi) * 0.05 + 1.0

    fig.update_layout(
        xaxis={"range": [idx.min(), x_end], "gridcolor": GRID,
               "rangeslider": {"visible": False}},
        yaxis={"title": "Price", "gridcolor": GRID,
               "range": [y_lo - pad, y_hi + pad], "autorange": False},
    )
    return _layout(fig, height, title)


def wave_vs_price_chart(
    close: pd.Series,
    wave: pd.Series,
    projection: Optional[pd.Series] = None,
    height: int = 460,
) -> go.Figure:
    """Normalised comparison, with an optional forward projection segment."""
    fig = go.Figure()
    c = close.dropna()
    if not c.empty:
        fig.add_trace(go.Scatter(
            x=c.index, y=np.log(c.values), mode="lines", name="log price",
            line={"color": "#E8E8E8", "width": 1.5},
        ))
    w = wave.dropna()
    if not w.empty:
        fig.add_trace(go.Scatter(
            x=w.index, y=w.values, mode="lines", name="alignment wave",
            line={"color": "#F2C14E", "width": 1.6}, yaxis="y2",
        ))
    if projection is not None:
        p = projection.dropna()
        if not p.empty:
            fig.add_trace(go.Scatter(
                x=p.index, y=p.values, mode="lines", name="projected wave",
                line={"color": "#F2C14E", "width": 1.6, "dash": "dot"}, yaxis="y2",
            ))
            fig.add_vline(x=c.index.max() if not c.empty else p.index.min(),
                          line_width=1, line_dash="dash",
                          line_color="rgba(200,200,200,0.4)")
    fig.update_layout(yaxis2={"title": "wave (z)", "overlaying": "y",
                              "side": "right", "showgrid": False})
    return _layout(fig, height)


def spectrum_chart(
    spec: pd.DataFrame,
    peaks: Optional[pd.DataFrame] = None,
    synodics: Optional[pd.DataFrame] = None,
    height: int = 420,
) -> go.Figure:
    """
    Periodogram on a log period axis, with detected peaks and synodic
    periods marked so overlaps are visible rather than asserted.
    """
    fig = go.Figure()
    if spec is None or spec.empty:
        return _layout(fig, height, "Not enough data for a spectrum")

    fig.add_trace(go.Scatter(
        x=spec["period_days"], y=spec["power"], mode="lines", name="power",
        line={"color": "#4FA8E0", "width": 1.4}, fill="tozeroy",
        fillcolor="rgba(79,168,224,0.18)",
    ))

    if peaks is not None and not peaks.empty:
        fig.add_trace(go.Scatter(
            x=peaks["period_days"], y=peaks["power"], mode="markers", name="peaks",
            marker={"size": 9, "color": "#F2C14E", "symbol": "circle-open",
                    "line": {"width": 2}},
            text=[f"{p:.0f}d" for p in peaks["period_days"]],
        ))

    if synodics is not None and not synodics.empty:
        lo, hi = float(spec["period_days"].min()), float(spec["period_days"].max())
        for _, r in synodics.iterrows():
            for h in (1, 2):
                p = float(r["synodic_days"]) / h
                if lo <= p <= hi:
                    fig.add_vline(
                        x=p, line_width=1, line_dash="dot",
                        line_color="rgba(230,150,90,0.45)",
                        annotation_text=f"{r['pair']}/{h}" if h > 1 else str(r["pair"]),
                        annotation_font_size=9,
                        annotation_textangle=-90,
                    )

    fig.update_layout(
        xaxis={"title": "period (calendar days)", "type": "log", "gridcolor": GRID},
        yaxis={"title": "spectral power", "gridcolor": GRID},
    )
    return _layout(fig, height)


def lead_lag_chart(ll: pd.DataFrame, height: int = 340) -> go.Figure:
    fig = go.Figure()
    if ll is None or ll.empty:
        return _layout(fig, height, "Not enough overlap for a lead/lag scan")
    fig.add_trace(go.Scatter(
        x=ll["lag_days"], y=ll["correlation"], mode="lines",
        line={"color": "#57B96C", "width": 1.8}, name="correlation",
    ))
    fig.add_hline(y=0, line_width=1, line_color="rgba(200,200,200,0.35)")
    j = int(ll["correlation"].abs().idxmax())
    fig.add_vline(x=float(ll.loc[j, "lag_days"]), line_width=1, line_dash="dash",
                  line_color="#F2C14E",
                  annotation_text=f"best {int(ll.loc[j,'lag_days'])}d",
                  annotation_font_size=10)
    fig.update_layout(
        xaxis={"title": "wave lead (days) - positive means wave leads price",
               "gridcolor": GRID},
        yaxis={"title": "correlation", "gridcolor": GRID},
    )
    return _layout(fig, height)


def event_study_chart(es: pd.DataFrame, height: int = 360) -> go.Figure:
    """Excess forward return per horizon, coloured by permutation p-value."""
    fig = go.Figure()
    if es is None or es.empty:
        return _layout(fig, height, "Not enough events for a study")
    colours = ["#57B96C" if p < 0.05 else "#7A7A7A" for p in es["p_permutation"]]
    fig.add_trace(go.Bar(
        x=es["horizon_days"].astype(str) + "d",
        y=es["excess"] * 100.0,
        marker_color=colours,
        text=[f"p={p:.3f}" for p in es["p_permutation"]],
        textposition="outside",
        name="excess return",
    ))
    fig.add_hline(y=0, line_width=1, line_color="rgba(200,200,200,0.4)")
    fig.update_layout(
        xaxis={"title": "horizon after alignment", "gridcolor": GRID},
        yaxis={"title": "excess return vs baseline (%)", "gridcolor": GRID},
        showlegend=False,
    )
    return _layout(fig, height)


def multi_line_chart(
    frame: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    height: int = 420,
    title: Optional[str] = None,
) -> go.Figure:
    fig = go.Figure()
    if frame is None or frame.empty:
        return _layout(fig, height, title or "Nothing to plot")
    cols = list(columns) if columns else list(frame.columns)
    for i, c in enumerate(cols):
        if c not in frame.columns:
            continue
        fig.add_trace(go.Scatter(
            x=frame.index, y=frame[c], mode="lines", name=str(c),
            line={"color": PALETTE[i % len(PALETTE)], "width": 1.3},
        ))
    return _layout(fig, height, title)


def sky_grid_chart(
    series: Dict[str, pd.Series],
    dots: pd.DataFrame,
    shapes: List[Dict],
    y_bounds: Tuple[float, float],
    extend_days: int = 180,
    height: int = 820,
    y_title: str = "Price",
    show_dot_lines: bool = True,
) -> go.Figure:
    """
    Overlaid price series, planetary peak-angle dots, and radiating rays.

    The y axis is pinned from `y_bounds` rather than left to autorange.
    Plotly folds shape coordinates into autorange, and with hundreds of rays
    any one of them escaping the band would stretch the axis and flatten
    every price series into a line along the bottom.
    """
    fig = go.Figure()
    if not series:
        return _layout(fig, height, "No data")

    x_min = min(s.index.min() for s in series.values())
    x_max = max(s.index.max() for s in series.values())
    x_end = x_max + pd.Timedelta(int(extend_days), "D") if extend_days else x_max

    for i, (ticker, s) in enumerate(series.items()):
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines", name=str(ticker),
            line={"color": PALETTE[i % len(PALETTE)], "width": 2},
        ))

    if dots is not None and not dots.empty:
        from .skygrid import PLANET_COLOURS
        for body, grp in dots.groupby("body"):
            colour = PLANET_COLOURS.get(str(body), "#9aa4b2")
            for ticker, sub in grp.groupby("ticker"):
                sub = sub.sort_values("date")
                fig.add_trace(go.Scatter(
                    x=sub["date"], y=sub["price"],
                    mode="lines+markers" if show_dot_lines else "markers",
                    name=f"{body} ({ticker})",
                    marker={"size": 7, "color": colour, "symbol": "circle",
                            "line": {"width": 1, "color": "#101418"}},
                    line={"color": colour, "width": 1, "dash": "dot"},
                    opacity=0.85,
                    customdata=np.stack([sub["angle"].to_numpy()], axis=-1),
                    hovertemplate=(f"<b>{body}</b> {ticker}<br>"
                                   "%{x|%Y-%m-%d}<br>"
                                   "angle %{customdata[0]:.2f} deg<br>"
                                   "mapped %{y:.2f}<extra></extra>"),
                ))

    if shapes:
        fig.update_layout(shapes=shapes)

    fig.update_layout(
        xaxis={"range": [x_min, x_end], "gridcolor": GRID,
               "rangeslider": {"visible": False}},
        yaxis={"title": y_title, "gridcolor": GRID,
               "range": [float(y_bounds[0]), float(y_bounds[1])],
               "autorange": False},
    )
    return _layout(fig, height)


def angle_track_chart(
    angles: pd.DataFrame,
    dots: Optional[pd.DataFrame] = None,
    domain: Tuple[float, float] = (0.0, 180.0),
    height: int = 320,
) -> go.Figure:
    """
    The raw sky angles in degrees, on their own axis.

    Kept separate from the price chart deliberately. On the price chart the
    angles are scaled by a per-stock ratio to make them comparable with an
    instrument, which means the numbers there are no longer degrees. This is
    where the actual measurement can be read.
    """
    fig = go.Figure()
    if angles is None or angles.empty:
        return _layout(fig, height, "No angle data")

    from .skygrid import PLANET_COLOURS
    for col in angles.columns:
        fig.add_trace(go.Scatter(
            x=angles.index, y=angles[col], mode="lines", name=str(col),
            line={"color": PLANET_COLOURS.get(str(col), "#9aa4b2"), "width": 1.2},
            opacity=0.75,
        ))

    if dots is not None and not dots.empty:
        for body, grp in dots.groupby("body"):
            g = grp.drop_duplicates(subset=["date"]).sort_values("date")
            fig.add_trace(go.Scatter(
                x=g["date"], y=g["angle"], mode="markers",
                name=f"{body} peak", showlegend=False,
                marker={"size": 8, "symbol": "circle-open",
                        "color": PLANET_COLOURS.get(str(body), "#9aa4b2"),
                        "line": {"width": 2}},
            ))

    fig.update_layout(
        yaxis={"title": "degrees", "gridcolor": GRID,
               "range": [float(domain[0]), float(domain[1])]},
        xaxis={"gridcolor": GRID},
    )
    return _layout(fig, height)


def degree_grid_chart(
    curves: Dict[str, pd.Series],
    dots: pd.DataFrame,
    shapes: List[Dict],
    now: pd.Timestamp,
    x_range: Tuple[pd.Timestamp, pd.Timestamp],
    domain: Tuple[float, float] = (0.0, 180.0),
    zones: Optional[pd.DataFrame] = None,
    height: int = 780,
    dot_size: int = 9,
) -> go.Figure:
    """
    Degree-space Gann grid.

    Vertical axis is degrees, not price. Planet dots sit at their measured
    angle with no conversion. The stock is scaled into the band as an
    overlay. Rays radiate from each dot in four directions, and marked
    confluence zones are where several of them cross.

    The axis is pinned to the domain. With hundreds of ray shapes, one
    escaping the band would drag the axis with it, since Plotly folds shape
    coordinates into autorange.
    """
    fig = go.Figure()

    if shapes:
        fig.update_layout(shapes=list(shapes))

    for i, (ticker, s) in enumerate(curves.items()):
        s = s.dropna()
        if s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines", name=str(ticker),
            line={"color": "#3DBE86" if i == 0 else PALETTE[i % len(PALETTE)],
                  "width": 2.2},
            hovertemplate=f"<b>{ticker}</b><br>%{{x|%Y-%m-%d}}<br>"
                          "%{y:.1f} deg<extra></extra>",
        ))

    if dots is not None and not dots.empty:
        from .skygrid import PLANET_COLOURS
        for body, grp in dots.groupby("body"):
            g = grp.sort_values("date")
            fig.add_trace(go.Scatter(
                x=g["date"], y=g["angle"], mode="markers",
                name=f"{body}",
                marker={"size": dot_size,
                        "color": PLANET_COLOURS.get(str(body), "#29B6F6"),
                        "line": {"width": 1, "color": "#0B1015"}},
                hovertemplate=f"<b>{body}</b><br>%{{x|%Y-%m-%d}}<br>"
                              "peak %{y:.2f} deg<extra></extra>",
            ))

    if zones is not None and not zones.empty:
        z = zones.head(40)
        fig.add_trace(go.Scatter(
            x=z["date"], y=z["degrees"], mode="markers",
            name="confluence",
            marker={"size": 9 + 2 * z["count"].clip(upper=6),
                    "symbol": "x", "color": "#F2C14E",
                    "line": {"width": 1, "color": "#F2C14E"}},
            customdata=np.stack([z["count"], z["days_from_now"]], axis=-1),
            hovertemplate=("confluence<br>%{x|%Y-%m-%d}<br>%{y:.1f} deg<br>"
                           "%{customdata[0]} crossings<br>"
                           "%{customdata[1]:+d} days from now<extra></extra>"),
        ))

    fig.add_vline(x=pd.Timestamp(now), line_width=2, line_color="#E0524B",
                  annotation_text="now", annotation_position="top",
                  annotation_font_color="#E0524B")

    fig.update_layout(
        xaxis={"range": [pd.Timestamp(x_range[0]), pd.Timestamp(x_range[1])],
               "gridcolor": GRID, "rangeslider": {"visible": False},
               "dtick": "M1", "tickformat": "%b %Y"},
        yaxis={"title": "degrees", "gridcolor": GRID,
               "range": [float(domain[0]), float(domain[1])],
               "autorange": False, "dtick": 30},
    )
    return _layout(fig, height)

"""
Kairos StarVector - main GUI
============================
Streamlit application. Launch from source with:

    python kairos_app.py

or via START.bat, or from the built executable.
"""
from __future__ import annotations

import os
import sys
import traceback
from datetime import date

# Make the project root importable whether this file is run from source,
# from a different working directory, or out of a PyInstaller bundle.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import streamlit as st

from kairos import astro as A
from kairos import calibrate as CAL
from kairos import charting as C
from kairos import gann as G
from kairos import skygrid as SG
from kairos import market as M
from kairos import paths as P
from kairos import waves as W

st.set_page_config(page_title="Kairos StarVector", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
      section[data-testid="stSidebar"] { min-width: 350px; max-width: 350px; }
      .kpi { padding: 10px 14px; border-radius: 10px;
             border: 1px solid rgba(255,255,255,0.12);
             background: rgba(255,255,255,0.035); }
      .kpi-t { font-size: 0.74rem; letter-spacing: .04em;
               text-transform: uppercase; color: #9aa4b2; }
      .kpi-v { font-size: 1.28rem; font-weight: 700; margin-top: 2px; }
      .kpi-s { font-size: 0.74rem; color: #8b95a3; }
      .note { border-left: 3px solid #4FA8E0; padding: 8px 12px;
              background: rgba(79,168,224,0.07); border-radius: 4px;
              font-size: 0.88rem; }
      .warn { border-left: 3px solid #E8973A; padding: 8px 12px;
              background: rgba(232,151,58,0.08); border-radius: 4px;
              font-size: 0.88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

def kpi(col, title: str, value: str, sub: str = "") -> None:
    col.markdown(
        f'<div class="kpi"><div class="kpi-t">{title}</div>'
        f'<div class="kpi-v">{value}</div>'
        f'<div class="kpi-s">{sub}</div></div>',
        unsafe_allow_html=True,
    )

def bounded_slider(container, label: str, lo: int, hi: int, default: int,
                   step: int = 1, **kw):
    """
    A slider that survives its own bounds collapsing.

    Streamlit raises StreamlitAPIException when min_value is not strictly
    less than max_value, and any bound computed from the data can collapse:
    "Anchors to include" ran from 2 to min(len(pivots), 40), so a chart with
    two detected pivots produced a slider from 2 to 2 and took the whole page
    down with it.

    When the range has no room, this reports the single available value as
    text and returns it. When it does, the default is clamped inside the
    range first, since a default outside the bounds is the same exception by
    another route.
    """
    lo, hi = int(lo), int(hi)
    if hi <= lo:
        container.caption(f"{label}: {lo} (only one value available here)")
        return lo
    return container.slider(label, lo, hi,
                            int(min(max(default, lo), hi)), step, **kw)


def bounded_number(container, label: str, lo: int, hi: int, default: int,
                   step: int = 1, **kw):
    """Integer number_input with the same collapse guard as bounded_slider."""
    lo, hi = int(lo), int(hi)
    if hi <= lo:
        container.caption(f"{label}: {lo}")
        return lo
    return container.number_input(label, lo, hi,
                                  int(min(max(default, lo), hi)), step, **kw)


# --------------------------------------------------------------------------
# Cached data layer
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Fetching price history...", ttl=3600, max_entries=16)
def load_prices(ticker: str, start: str, end: str, interval: str, demo: bool) -> pd.DataFrame:
    if demo:
        return M.demo_ohlc(start, end)
    return M.fetch_ohlc(ticker, start=start, end=end, interval=interval)

@st.cache_data(show_spinner="Computing planetary positions...", max_entries=16)
def load_longitudes(start: str, end: str, bodies: tuple, frame: str) -> pd.DataFrame:
    idx = A.calendar_index(start, end)
    return A.longitudes(idx, bodies=list(bodies), frame=frame)

@st.cache_data(show_spinner="Finding alignments...", max_entries=32)
def load_events(lons_key: tuple, aspects: tuple, orb: float, min_sep: int,
                pairs: tuple, _lons: pd.DataFrame) -> pd.DataFrame:
    spec = A.EventSpec(aspects=list(aspects), orb_deg=float(orb),
                       min_separation_days=int(min_sep),
                       pairs=[tuple(p) for p in pairs] if pairs else None)
    ev = A.aspect_events(_lons, spec)
    if not ev.empty:
        ev["label"] = ev["pair"] + " " + ev["aspect_name"]
    return ev

@st.cache_data(show_spinner="Measuring synodic periods...", max_entries=8)
def load_synodics(bodies: tuple) -> pd.DataFrame:
    return A.synodic_periods(bodies=list(bodies))

@st.cache_data(show_spinner=False, max_entries=32)
def load_stations(lons_key: tuple, _lons: pd.DataFrame) -> pd.DataFrame:
    return A.station_events(_lons)

def frame_key(df: pd.DataFrame) -> tuple:
    """Cheap cache key for a DataFrame without hashing every cell."""
    if df is None or df.empty:
        return ("empty",)
    return (str(df.index.min()), str(df.index.max()), len(df), tuple(df.columns))

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("Kairos StarVector")
    st.caption("Market geometry - planetary alignment - Gann structure")

    with st.expander("Market data", expanded=True):
        demo_mode = st.checkbox(
            "Offline demo data", value=False,
            help="Synthetic random walk. Also the control case: it has no "
                 "relationship to planetary geometry by construction, so "
                 "whatever the analysis tabs report here is your noise floor.",
        )
        ticker = st.text_input("Ticker (Yahoo symbol)", value="^GSPC",
                               disabled=demo_mode)
        today = date.today()
        c1, c2 = st.columns(2)
        start_d = c1.date_input("Start", value=date(2015, 1, 1),
                                min_value=date(1928, 1, 1), max_value=today)
        end_d = c2.date_input("End", value=today,
                              min_value=date(1928, 1, 2),
                              max_value=date(today.year + 1, 12, 31))
        interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0,
                                disabled=demo_mode)

    with st.expander("Planetary geometry", expanded=True):
        frame = st.radio(
            "Frame", ["geocentric", "heliocentric"], index=0, horizontal=True,
            help="Geocentric is what Gann and astrologers used. Heliocentric "
                 "is what the Bradley siderograph uses; Earth replaces the Sun.",
        )
        avail = A.HELIO_BODIES if frame.startswith("helio") else A.BODIES
        default_bodies = [b for b in
                          ["SUN", "EARTH", "VENUS", "MARS", "JUPITER", "SATURN", "URANUS"]
                          if b in avail]
        bodies = st.multiselect("Bodies", avail, default=default_bodies)

        aspects = st.multiselect(
            "Aspect angles", A.ASPECTS_ALL, default=[0.0, 90.0, 120.0, 180.0],
            format_func=lambda a: f"{a:g}  {A.aspect_label(a)}",
        )
        orb = st.slider("Orb (degrees)", 0.5, 8.0, 2.0, 0.5,
                        help="Half-width of the window counted as an aspect. "
                             "Wider orbs find more events but each is weaker.")
        min_sep = st.slider("Minimum days between events", 1, 30, 3)

        pair_mode = st.radio("Pairs", ["All combinations", "Choose one pair"],
                             index=0, horizontal=True)
        chosen_pairs = ()
        if pair_mode == "Choose one pair" and len(bodies) >= 2:
            pc1, pc2 = st.columns(2)
            pa = pc1.selectbox("Body A", bodies, index=0)
            pb = pc2.selectbox("Body B", bodies,
                               index=min(1, len(bodies) - 1))
            if pa != pb:
                chosen_pairs = ((pa, pb),)

    with st.expander("Trickle-down wave", expanded=True):
        st.caption(
            "Each alignment emits an influence that decays forward in time. "
            "Overlapping tails from many events sum into a wave."
        )
        tau = st.slider("Decay constant tau (days)", 2, 365, 30,
                        help="Influence falls to 37% of its starting value "
                             "after this many days.")
        horizon = st.slider("Forward reach (days)", 10, 900, 180,
                            help="How far an event's influence is allowed to spread.")
        lead_days = st.slider("Anticipation before event (days)", 0, 120, 0,
                              help="Non-zero if you think positioning happens "
                                   "ahead of the alignment.")
        use_osc = st.checkbox("Oscillating tail", value=False,
                              help="Adds a cosine to the decay so a single event "
                                   "produces a wave rather than a bump.")
        osc_period = st.slider("Oscillation period (days)", 5, 720, 90,
                               disabled=not use_osc)
        osc_phase = st.slider("Phase (degrees)", 0, 359, 0, disabled=not use_osc)
        weight_by = st.selectbox(
            "Event weighting",
            ["strength (body + aspect + exactness)", "equal"], index=0,
        )

    with st.expander("Gann overlay", expanded=False):
        show_gann = st.checkbox("Draw Gann geometry", value=True)
        fan_labels = st.multiselect(
            "Fan ratios", list(G.FAN_RATIOS.keys()), default=["1x1"],
            help="Start with 1x1 alone. Each extra ratio multiplies the line "
                 "count by the number of anchors, and the ratios are near "
                 "parallel to each other, so three ratios across many anchors "
                 "reads as a wash rather than as structure.")
        unit_mode = st.radio("1x1 unit", ["Auto (swing rate)", "Manual"],
                             index=0, horizontal=True,
                             help="Auto uses the median price change per day "
                                  "measured pivot to pivot. Median daily "
                                  "change was the first attempt and is far too "
                                  "steep: it is noise rate, not trend rate, "
                                  "and it projects the fan off the chart.")
        manual_unit = st.number_input("Price per day", value=1.0, step=0.1,
                                      min_value=0.0001, format="%.4f",
                                      disabled=unit_mode.startswith("Auto"))
        anchor_mode = st.radio(
            "Fan anchors at", ["alignments", "pivots", "both"], index=0,
            horizontal=True,
            help="Alignments puts a fan on every planetary alignment. Pivots "
                 "puts them on swing highs and lows, which is the classical "
                 "Gann placement.")
        align_stride = st.number_input(
            "Alignment interval (every Nth)", 1, 50, 1, 1,
            help="1 draws a fan on every alignment, 3 on every third, and so "
                 "on. Use this to thin the chart rather than lowering the "
                 "anchor cap, which would just truncate the series early.")
        ray_length = st.slider(
            "Ray length (days)", 30, 1000, 120, 10,
            help="How far each ray runs from its origin. This is the single "
                 "biggest control over readability. Left to its automatic "
                 "value a 1x1 ray covers the entire price range, so every "
                 "anchor draws a line across the whole chart and 40 anchors "
                 "become an opaque lattice.")
        pivot_window = st.slider("Pivot detection window (bars)", 5, 120, 21)
        max_anchors = st.slider("Max fan anchors", 1, 200, 8,
                                help="A fan is 9 ratios times 2 directions, "
                                     "so 40 anchors is roughly 700 lines. "
                                     "Past a few hundred the chart is a grey "
                                     "wash and Plotly slows down.")
        both_dirs = st.checkbox("Fans both directions", value=False,
                                help="Off by default now. With a fan on every "
                                     "alignment, mirroring doubles the line "
                                     "count for little added information.")
        st.markdown("---")
        use_sq9 = st.checkbox("Square of Nine levels", value=True)
        sq9_base_mode = st.radio("SQ9 base", ["Last close", "Lowest low", "Manual"],
                                 index=0, horizontal=True, disabled=not use_sq9)
        sq9_manual = st.number_input("Base price", value=100.0, step=1.0,
                                     disabled=(not use_sq9 or sq9_base_mode != "Manual"))
        sq9_rot = st.slider("SQ9 rotations", 1, 5, 2, disabled=not use_sq9)
        round_step = st.number_input("Round-number grid step (0 = off)",
                                     value=0.0, step=1.0, min_value=0.0)
        show_time_cycles = st.checkbox("Gann time counts from top pivot", value=False)
        show_verticals = st.checkbox("Vertical lines at alignments", value=True)
        extend_days = st.slider("Extend chart forward (days)", 0, 720, 180)

    with st.expander("Planetary price lines", expanded=False):
        show_planet_lines = st.checkbox("Convert longitude to price", value=False)
        pl_bodies = st.multiselect("Bodies to convert",
                                   [b for b in bodies], default=[])
        pl_mode = st.radio("Conversion", ["sq9", "scale"], index=0, horizontal=True,
                           help="sq9 maps degrees onto the Square of Nine spiral. "
                                "scale multiplies degrees by a constant.")
        pl_scale = st.number_input("Degrees to price factor", value=1.0, step=0.1,
                                   disabled=(pl_mode != "scale"))
        pl_harmonic = st.slider("Harmonic shift (revolutions)", -3, 3, 0)
        pl_autofit = st.checkbox("Auto-centre on price", value=True)

    with st.expander("Statistics", expanded=False):
        n_surrogates = st.slider("Surrogate iterations", 50, 2000, 300, 50,
                                 help="More is slower but gives a finer p-value.")
        n_permutations = st.slider("Event-study permutations", 100, 5000, 500, 100)
        max_lag = st.slider("Lead/lag scan range (days)", 10, 400, 120)
        horizons_txt = st.text_input("Event horizons (days, comma separated)",
                                     value="1,3,5,10,21")
        reality_check = st.checkbox(
            "Reality check on shuffled price", value=False,
            help="Re-runs every test against a phase-randomised version of "
                 "your own price series, which has the same volatility and "
                 "autocorrelation but no real relationship to anything. "
                 "Whatever it reports is the score chance produces.",
        )

    st.markdown("---")
    run = st.button("Run analysis", type="primary", width="stretch")

# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------
if not bodies or len(bodies) < 2:
    st.info("Select at least two bodies in the sidebar, then press Run analysis.")
    st.stop()
if not aspects:
    st.info("Select at least one aspect angle in the sidebar.")
    st.stop()
if start_d >= end_d:
    st.error("Start date must be before end date.")
    st.stop()

if not run and "loaded" not in st.session_state:
    st.title("Kairos StarVector")
    st.markdown(
        """
        Aligns market price history against planetary geometry and Gann
        structure, then measures whether the alignment actually relates to
        price rather than assuming it does.

        **What each tab does**

        | Tab | Purpose |
        |---|---|
        | Chart | Price with Gann fans, Square of Nine levels, planetary price lines, and the alignment wave |
        | Alignment wave | The trickle-down model: events convolved with a decay kernel, plus a forward projection |
        | Statistics | Correlations and event studies with autocorrelation-aware p-values |
        | Cycles | Periods actually present in price, matched against synodic periods |
        | Calendar | Every alignment, station and ingress, past and future |
        | Forecast models | Optional LSTM / cWGAN benchmark, if the ML extras are installed |

        Set your parameters in the sidebar and press **Run analysis**.
        """
    )
    st.markdown(
        '<div class="note"><b>On prediction.</b> Planetary positions are '
        'deterministic and this program computes them exactly, decades ahead if '
        'you want. Whether those positions relate to price is a separate, '
        'empirical question, and it is the question the Statistics tab exists to '
        'answer. The tests there are built to be hard to pass: they compare '
        'against surrogate series that share your data\'s volatility and '
        'autocorrelation. Naive correlations between two smooth series routinely '
        'exceed 0.8 with no relationship whatsoever, so the app reports the '
        'naive and the honest number side by side. Read both.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

st.session_state["loaded"] = True

# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------
start_s, end_s = start_d.isoformat(), end_d.isoformat()

try:
    df = load_prices(ticker, start_s, end_s, interval, demo_mode)
except Exception as exc:
    st.error(f"Could not load price data: {exc}")
    st.markdown(
        '<div class="warn">Tick <b>Offline demo data</b> in the sidebar to '
        'explore the app without a connection.</div>', unsafe_allow_html=True)
    st.stop()

if df.empty or len(df) < 40:
    st.error(f"Only {len(df)} usable bars returned. Widen the date range or check the symbol.")
    st.stop()

close = df["Close"]
first, last = close.index.min(), close.index.max()

lons_full = load_longitudes(
    first.strftime("%Y-%m-%d"),
    (last + pd.Timedelta(int(max(extend_days, 400)), "D")).strftime("%Y-%m-%d"),
    tuple(bodies), frame,
)
lons_hist = lons_full.loc[:last]

events = load_events(frame_key(lons_full), tuple(aspects), float(orb),
                     int(min_sep), tuple(chosen_pairs), lons_full)
events_hist = events[events["date"] <= last].copy() if not events.empty else events.copy()
events_future = events[events["date"] > last].copy() if not events.empty else events.copy()

if weight_by == "equal" and not events.empty:
    events = events.assign(weight=1.0)
    events_hist = events_hist.assign(weight=1.0)
    events_future = events_future.assign(weight=1.0)

wave_full = W.composite_pressure(
    lons_full.index, events,
    tau_days=float(tau), horizon_days=int(horizon),
    period_days=float(osc_period) if use_osc else None,
    phase_deg=float(osc_phase) if use_osc else 0.0,
    lead_days=int(lead_days),
)
wave_hist = wave_full.loc[:last]
wave_future = wave_full.loc[last:]

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("Kairos StarVector")
info = M.describe(df)
cols = st.columns(6)
kpi(cols[0], "Symbol", "DEMO" if demo_mode else ticker,
    f"{info['bars']} bars, {interval}")
kpi(cols[1], "Range", f"{info['first']}", f"to {info['last']}")
kpi(cols[2], "Last close", f"{info['last_close']:,.2f}",
    f"{info['total_return_pct']:+.1f}% over span")
kpi(cols[3], "Alignments", f"{len(events_hist)}",
    f"{len(events_future)} ahead of last bar")
kpi(cols[4], "Wave range", f"{wave_hist.min():.2f} to {wave_hist.max():.2f}",
    f"tau={tau}d, reach={horizon}d")
kpi(cols[5], "Frame", frame[:5], f"{len(bodies)} bodies")

if demo_mode:
    st.markdown(
        '<div class="warn"><b>Demo data.</b> This is a synthetic random walk with '
        'no connection to planetary geometry. Any pattern the analysis tabs find '
        'here is chance. That makes it a useful yardstick for the settings you '
        'are using.</div>', unsafe_allow_html=True)

tabs = st.tabs(["Gann sky grid", "Chart", "Alignment wave", "Statistics",
                "Calibrate & forecast", "Cycles", "Calendar",
                "Forecast models", "Data"])

# --------------------------------------------------------------------------
# Chart
# --------------------------------------------------------------------------
with tabs[1]:
    shapes, meta = [], {}
    if show_gann:
        pivots = G.find_pivots(close, window=int(pivot_window),
                               min_separation=max(int(pivot_window) // 2, 3))
        if sq9_base_mode == "Last close":
            sq9_base = float(close.iloc[-1])
        elif sq9_base_mode == "Lowest low":
            sq9_base = float(close.min())
        else:
            sq9_base = float(sq9_manual)

        shapes, meta = G.build_overlay(
            close,
            pivots=pivots,
            alignment_dates=events_hist["date"].tolist() if not events_hist.empty else None,
            ratios=[G.FAN_RATIOS[k] for k in fan_labels] or [1.0],
            unit_price_per_day=None if unit_mode.startswith("Auto") else float(manual_unit),
            extend_days=int(extend_days),
            max_fan_anchors=int(max_anchors),
            both_directions=bool(both_dirs),
            anchor_mode=anchor_mode,
            alignment_stride=int(align_stride),
            sq9_base=sq9_base if use_sq9 else None,
            sq9_rotations=int(sq9_rot),
            round_step=float(round_step),
            show_time_cycles=bool(show_time_cycles),
            show_alignment_verticals=bool(show_verticals),
            max_days=int(ray_length),
        )

    planet_lines = {}
    if show_planet_lines and pl_bodies:
        for b in pl_bodies:
            series = G.planet_price_series(
                lons_hist, b, mode=pl_mode,
                deg_to_price=float(pl_scale),
                base_price=float(close.iloc[0]),
                harmonic=int(pl_harmonic),
            )
            if pl_autofit and not series.empty:
                series = series + G.fit_offset_to_price(series, close)
            planet_lines[f"{b} price"] = series

    marker_src = events_hist
    if len(marker_src) > 250:
        marker_src = marker_src.nlargest(250, "weight")

    fig = C.price_chart(
        df, shapes=shapes, planet_lines=planet_lines,
        wave=wave_full if extend_days else wave_hist,
        wave_name="Alignment wave (z)",
        event_markers=marker_src,
        extend_days=int(extend_days),
        candles=(interval == "1d" and len(df) <= 4000),
    )
    st.plotly_chart(fig, width="stretch", theme=None)

    if meta:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("1x1 unit (price/day)", f"{meta.get('unit_price_per_day', 0):.4f}")
        m2.metric("Fan anchors", len(meta.get("fan_anchors", [])))
        m3.metric("Fan horizon", f"{meta.get('fan_horizon_days', 0)} d")
        m4.metric("SQ9 levels in view", meta.get("sq9_levels_in_view", 0))
        m5.metric("Total shapes", len(shapes))
        st.caption(
            f"Price axis pinned to {meta.get('y_bounds')} and every line "
            f"clipped to it. Fan rays also stop "
            f"{meta.get('fan_horizon_days', 0)} days past their anchor, the "
            f"point at which a 1x1 at this rate has left the chart: beyond it "
            f"price sits below every ratio, which reads as an extremely weak "
            f"market when it only means the fan expired."
        )
        with st.expander("Anchor detail"):
            if meta.get("fan_anchors"):
                st.dataframe(pd.DataFrame(meta["fan_anchors"]),
                             width="stretch", hide_index=True)
            if meta.get("time_cycle_dates"):
                st.write("Time counts from", meta.get("time_cycle_anchor"))
                st.write(", ".join(meta["time_cycle_dates"]))

    st.download_button(
        "Download chart as standalone HTML",
        data=fig.to_html(include_plotlyjs="cdn", full_html=True),
        file_name=f"kairos_{'demo' if demo_mode else ticker}_chart.html",
        mime="text/html",
    )

    # ---- Gann angle analysis ------------------------------------------
    with st.expander("Gann angle analysis", expanded=False):
        st.caption(
            "Four claims from the Gann angle literature, implemented as "
            "measurements rather than drawings: market strength read from "
            "which angle price trades above, the rule that a broken angle "
            "leads price to the next one, price clusters where angles from "
            "different anchors converge, and angle crossings of retracement "
            "levels."
        )
        pv_all = G.find_pivots(close, window=int(pivot_window),
                               min_separation=max(int(pivot_window) // 2, 3))
        if pv_all.empty or len(close) < 60:
            st.info("Not enough pivots detected. Lower the pivot window.")
        else:
            g_unit_raw = G.auto_unit_price_per_day(close, pv_all)
            gu1, gu2, gu3, gu4 = st.columns(4)
            snap = gu1.checkbox("Harmonise 1x1", value=False,
                                help="Snap the rate to a value that divides "
                                     "360 evenly, the way a practitioner "
                                     "rounds 59 to 60.")
            g_unit = G.harmonise_unit(g_unit_raw) if snap else g_unit_raw
            g_horizon = G.fan_horizon_days(close, g_unit)
            gu2.metric("1x1 rate", f"{g_unit:.4f}", "price per day")
            gu3.metric("Useful horizon", f"{g_horizon} d",
                       "before lines leave the chart")

            labels = [f"{r['date'].date()}  {r['kind']}  {r['price']:.2f}"
                      for _, r in pv_all.iterrows()]
            default_i = len(labels) - 1
            for i in range(len(pv_all) - 1, -1, -1):
                if (last - pd.Timestamp(pv_all.iloc[i]["date"])).days >= 60:
                    default_i = i
                    break
            pick_a = gu4.selectbox("Anchor pivot", range(len(labels)),
                                   index=default_i,
                                   format_func=lambda i: labels[i])
            arow = pv_all.iloc[int(pick_a)]
            a_dir = 1 if arow["kind"] == "low" else -1

            st.markdown("###### Market strength relative to the fan")
            state = G.angle_state(close, arow["date"], arow["price"], g_unit,
                                  direction=a_dir, max_days=g_horizon)
            live = state.dropna(subset=["above"])
            if live.empty:
                st.info("This anchor is outside its useful horizon. Pick a "
                        "more recent pivot.")
            else:
                cur = live.iloc[-1]
                sa, sb, sc = st.columns(3)
                sa.metric("Current band", str(cur["band"]))
                sb.metric("Strength", f"{cur['strength']:.2f}", "0 weak, 1 strong")
                sc.metric("Distance from 1x1", f"{cur['dist_1x1']:+.2f}")
                st.line_chart(live[["strength"]], height=180)

                st.markdown("###### Rule of all angles")
                rb1, rb2 = st.columns([1, 3])
                horiz = rb1.number_input("Bars allowed to reach target",
                                         5, 250, 60, 5, key="roaa_bars")
                ev_g, sm = G.rule_of_all_angles(
                    close, arow["date"], arow["price"], g_unit,
                    direction=a_dir, max_bars_to_target=int(horiz),
                    max_days=g_horizon)
                if sm["n_events"] == 0:
                    rb2.info("No angle breaks found for this anchor.")
                else:
                    e1, e2, e3, e4 = st.columns(4)
                    e1.metric("Breaks", sm["n_events"])
                    e2.metric("Reached next angle", f"{sm['hit_rate']:.1%}")
                    e3.metric("Matched control",
                              f"{sm['control_hit_rate']:.1%}"
                              if pd.notna(sm["control_hit_rate"]) else "n/a")
                    e4.metric("Edge",
                              f"{sm['edge']:+.1%}"
                              if pd.notna(sm["edge"]) else "n/a")
                    st.caption(
                        f"The control uses the same break dates and the same "
                        f"initial distance but a static target instead of a "
                        f"fan line. Read the edge, not the hit rate: in any "
                        f"drifting market price reaches the next level up "
                        f"simply by drifting. Median {sm['median_bars']:.0f} "
                        f"bars to target when reached.<br><br>"
                        f"A negative edge on upward breaks is partly "
                        f"mechanical rather than evidence against the rule: a "
                        f"rising fan line recedes as time passes, so it is "
                        f"strictly harder to reach than a fixed level the "
                        f"same distance away, while a falling line descends "
                        f"to meet price. Compare up and down separately in "
                        f"the table.",
                        unsafe_allow_html=True)
                    st.dataframe(ev_g.tail(60), width="stretch",
                                 hide_index=True, height=240)
                    st.download_button("Download angle breaks CSV",
                                       ev_g.to_csv(index=False).encode(),
                                       file_name="kairos_angle_breaks.csv",
                                       mime="text/csv")

            st.markdown("###### Price clusters")
            lo_w = float(close.min()) * 0.85
            hi_w = float(close.max()) * 1.15
            n_anch = bounded_slider(st, "Anchors to include", 2,
                                    min(len(pv_all), 40), 12, 1, key="clus_n")
            anchors = [(r["date"], r["price"], 1 if r["kind"] == "low" else -1)
                       for _, r in pv_all.tail(int(n_anch)).iterrows()]
            clus = G.price_clusters(anchors, last, g_unit, min_count=2,
                                    price_window=(lo_w, hi_w),
                                    max_days=g_horizon)
            if clus.empty:
                st.info("No clusters at this bar. Include more anchors.")
            else:
                st.dataframe(clus.head(15), width="stretch", hide_index=True)
                st.caption(
                    "Counts are only comparable between zones computed from "
                    "the same anchor and ratio set: adding anchors raises "
                    "every count mechanically, not because the market moved."
                )

            st.markdown("###### Retracements and squaring")
            rc1, rc2 = st.columns(2)
            swing_hi = float(close.loc[arow["date"]:].max())
            swing_lo = float(close.loc[arow["date"]:].min())
            rets = G.retracement_levels(swing_hi, swing_lo)
            with rc1:
                st.caption(f"Eighths and thirds of {swing_lo:.2f} to {swing_hi:.2f}")
                st.dataframe(rets, width="stretch", hide_index=True, height=200)
            with rc2:
                sq = G.square_the_range(arow["date"], swing_hi, swing_lo, g_unit)
                st.caption("Range squared with time, from the anchor")
                st.dataframe(sq, width="stretch", hide_index=True, height=200)
            conf = G.angle_retracement_confluence(
                pd.date_range(arow["date"], last + pd.Timedelta(int(extend_days), "D"),
                              freq="D"),
                arow["date"], arow["price"], g_unit, rets, direction=a_dir)
            if not conf.empty:
                st.caption("Angle crossings of retracement levels, "
                           "computable in advance since both are geometric")
                st.dataframe(conf.tail(20), width="stretch", hide_index=True)

# --------------------------------------------------------------------------
# Alignment wave
# --------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Events convolved with a decay kernel")
    st.markdown(
        f"""
        Each of the **{len(events)}** alignments in range contributes a pulse that
        decays with a {tau}-day constant and reaches {horizon} days forward
        {'with an oscillating tail of ' + str(osc_period) + ' days' if use_osc else 'without oscillation'}.
        Overlapping tails sum, so clusters of alignments build a larger
        amplitude than isolated ones. The result is z-scored.
        """
    )

    lags, kern = W.trickle_kernel(int(horizon), float(tau),
                                  float(osc_period) if use_osc else None,
                                  float(osc_phase) if use_osc else 0.0,
                                  int(lead_days))
    kdf = pd.DataFrame({"kernel": kern}, index=pd.Index(lags, name="days from event"))
    kc1, kc2 = st.columns([1, 2])
    with kc1:
        st.caption("Single-event impulse response")
        st.line_chart(kdf, height=260)
    with kc2:
        st.caption("Composite wave against price")
        st.plotly_chart(
            C.wave_vs_price_chart(close, wave_hist,
                                  wave_future if extend_days else None),
            width="stretch",
            theme=None,
        )

    if extend_days and not wave_future.empty:
        st.markdown(
            '<div class="note"><b>The dotted segment is a projection of geometry, '
            'not of price.</b> Future planetary positions are exactly computable, '
            'so the wave beyond the last bar is certain in a way no price forecast '
            'can be. Whether the wave has anything to say about price is what the '
            'Statistics tab measures.</div>', unsafe_allow_html=True)

        peaks = wave_future[
            (wave_future.shift(1) < wave_future) & (wave_future.shift(-1) < wave_future)
        ].nlargest(10)
        troughs = wave_future[
            (wave_future.shift(1) > wave_future) & (wave_future.shift(-1) > wave_future)
        ].nsmallest(10)
        pc1, pc2 = st.columns(2)
        pc1.caption("Projected wave peaks")
        pc1.dataframe(
            pd.DataFrame({"date": peaks.index.date, "wave (z)": peaks.values.round(3)}),
            width="stretch", hide_index=True)
        pc2.caption("Projected wave troughs")
        pc2.dataframe(
            pd.DataFrame({"date": troughs.index.date, "wave (z)": troughs.values.round(3)}),
            width="stretch", hide_index=True)

    st.markdown("---")
    st.subheader("Continuous harmonic index")
    st.caption(
        "An alternative to discrete events: the sum of cos(h x separation) over "
        "all pairs. No orb, no threshold, nothing discarded. Harmonic 1 peaks on "
        "conjunctions, 2 adds oppositions, 4 adds squares."
    )
    h_sel = st.multiselect("Harmonics", [1, 2, 3, 4, 6, 8], default=[1, 2, 4])
    if h_sel:
        hframe = pd.DataFrame(
            {f"h{h}": A.harmonic_index(lons_hist, harmonic=int(h)) for h in h_sel}
        )
        st.plotly_chart(C.multi_line_chart(hframe), width="stretch", theme=None)

# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Does the wave relate to price?")
    st.markdown(
        '<div class="note">Price and the wave are both smooth and strongly '
        'autocorrelated. A plain correlation between two such series is almost '
        'always large, and its textbook p-value is wrong by orders of magnitude: '
        'two independent random walks correlate above 0.8 routinely. Every test '
        'below therefore reports a <b>surrogate</b> p-value computed against '
        'series that preserve your data\'s own spectrum and autocorrelation but '
        'carry no relationship. The naive p-value is shown next to it so you can '
        'see the size of the gap.</div>', unsafe_allow_html=True)

    try:
        horizons = tuple(int(x.strip()) for x in horizons_txt.split(",")
                         if x.strip().isdigit()) or (1, 5, 21)
    except Exception:
        horizons = (1, 5, 21)

    with st.spinner("Running surrogate and permutation tests..."):
        summary = W.summarise_relationship(
            close, wave_hist, events_hist,
            max_lag_days=int(max_lag), n_surrogates=int(n_surrogates),
            horizons=horizons, n_permutations=int(n_permutations),
        )

    def stat_row(label: str, d: dict) -> dict:
        return {
            "comparison": label,
            "n": d.get("n"),
            "pearson r": None if pd.isna(d.get("pearson", np.nan)) else round(d["pearson"], 4),
            "spearman": None if pd.isna(d.get("spearman", np.nan)) else round(d["spearman"], 4),
            "p (naive)": None if pd.isna(d.get("p_naive", np.nan)) else f"{d['p_naive']:.2e}",
            "p (surrogate)": None if pd.isna(d.get("p_surrogate", np.nan)) else round(d["p_surrogate"], 4),
            "survives": ("yes" if d.get("p_surrogate", 1) < 0.05 else "no"),
        }

    st.dataframe(pd.DataFrame([
        stat_row("wave vs log price (level)", summary["level_correlation"]),
        stat_row("wave vs daily return", summary["return_correlation"]),
        stat_row("wave vs forward 21d return", summary["forward_21d_correlation"]),
    ]), width="stretch", hide_index=True)

    st.markdown("#### Lead and lag")
    st.caption(
        "Positive lag means the wave moves before price, which is the shape the "
        "trickle-down idea predicts. Scanning every lag and keeping the largest "
        "correlation inflates it, so treat the peak as a hypothesis to test, not "
        "a result."
    )
    lc1, lc2 = st.columns([3, 1])
    lc1.plotly_chart(C.lead_lag_chart(summary["lead_lag"]), width="stretch", theme=None)
    lc2.metric("Best lag", f"{summary['best_lag_days']} d"
               if pd.notna(summary["best_lag_days"]) else "n/a")
    lc2.metric("Correlation there",
               f"{summary['best_lag_correlation']:.4f}"
               if pd.notna(summary["best_lag_correlation"]) else "n/a")
    ll_df = summary["lead_lag"]
    effective = int(ll_df["lag_days"].abs().max()) if not ll_df.empty else 0
    if effective < int(max_lag):
        lc2.caption(
            f"Requested +/-{max_lag} days, scanned +/-{effective}. "
            f"The loaded history cannot support a wider shift and still leave "
            f"32 overlapping days to correlate.")
    else:
        lc2.caption(f"Scanned +/-{effective} days")

    st.markdown("#### Forward returns after alignments")
    es = summary["event_study"]
    if es is None or es.empty:
        st.info("Not enough events in range for an event study.")
    else:
        st.plotly_chart(C.event_study_chart(es), width="stretch", theme=None)
        st.dataframe(es, width="stretch", hide_index=True)
        st.caption(
            f"Excess is the mean forward return minus the all-days baseline. "
            f"p is the share of {n_permutations} random date sets of the same "
            f"size that beat it. With {len(es)} horizons tested, roughly "
            f"{max(1, round(0.05 * len(es), 2))} false positive at p<0.05 is "
            f"expected by chance, so look for consistency across horizons "
            f"rather than the single best cell."
        )

    if reality_check:
        st.markdown("---")
        st.subheader("Reality check")
        st.caption(
            "The same tests against a phase-randomised copy of your price series. "
            "It keeps your volatility and autocorrelation but destroys any real "
            "relationship. Treat its numbers as the score chance produces with "
            "these settings."
        )
        with st.spinner("Running the null case..."):
            rng = np.random.default_rng(1234)
            daily = W.to_calendar_daily(close)
            logp = np.log(daily.to_numpy(dtype=float))
            surr = W._phase_randomise(W.detrend(logp, "linear"), rng)
            fake = pd.Series(np.exp(surr - surr.mean() + np.log(float(close.mean()))),
                             index=daily.index, name="Close")
            null_summary = W.summarise_relationship(
                fake, wave_hist, events_hist,
                max_lag_days=int(max_lag),
                n_surrogates=max(int(n_surrogates) // 2, 50),
                horizons=horizons,
                n_permutations=max(int(n_permutations) // 2, 100),
            )
        st.dataframe(pd.DataFrame([
            stat_row("NULL: wave vs log price", null_summary["level_correlation"]),
            stat_row("NULL: wave vs daily return", null_summary["return_correlation"]),
            stat_row("NULL: wave vs fwd 21d", null_summary["forward_21d_correlation"]),
        ]), width="stretch", hide_index=True)
        if not null_summary["event_study"].empty:
            st.dataframe(null_summary["event_study"],
                         width="stretch", hide_index=True)

# --------------------------------------------------------------------------
# Calibrate and forecast
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Precomputing geometry for the search...", max_entries=8)
def _precompute(start: str, end: str, bodies: tuple, aspects: tuple,
                max_orb: float, frame: str):
    return CAL.precompute(start, end, list(bodies), list(aspects),
                          max_orb=float(max_orb), frame=frame)

with tabs[4]:
    st.subheader("Fit on history, verify out of sample, then project forward")
    st.markdown(
        """
        1. Pick a split date. Everything before it is **train**, everything
           after is **test**.
        2. The search tries hundreds of parameter combinations, scoring each
           one on train only.
        3. The winners are re-scored on test, which the search never saw.
        4. Apply the parameters you choose past the last price bar to get
           dated positions.

        Step 3 is what makes steps 1 and 2 worth anything. A search over
        hundreds of combinations will always find something that fits the
        training window; that is what searching does. Read the **test** column
        and the **gap**, not the train score.
        """
    )

    # The split bounds are derived from bar positions, not calendar days.
    # search() needs at least MIN_TRAIN bars before the split and MIN_TEST
    # after it, and only the bar count can express that: a calendar-day
    # margin says nothing about how many trading days it contains, and on a
    # short range the two margins can cross and produce min_value >
    # max_value, which is a hard Streamlit error rather than a warning.
    MIN_TRAIN, MIN_TEST = 100, 60
    n_bars = len(close)

    if n_bars < MIN_TRAIN + MIN_TEST + 10:
        st.markdown(
            f'<div class="warn"><b>This range has {n_bars} bars; calibration '
            f'needs at least {MIN_TRAIN + MIN_TEST + 10}.</b><br><br>'
            f'The search fits parameters on a training window and checks them '
            f'on a held-out test window, which takes {MIN_TRAIN} bars before '
            f'the split and {MIN_TEST} after it as a bare minimum. Splitting '
            f'{n_bars} bars leaves too little on one side to mean anything.'
            f'<br><br>Widen the date range in the sidebar. About three years '
            f'of daily bars is a sensible floor, and more is better: the point '
            f'of the test window is to cover market conditions the training '
            f'window did not.</div>',
            unsafe_allow_html=True)
    else:
        split_lo = close.index[MIN_TRAIN].date()
        split_hi = close.index[n_bars - MIN_TEST - 1].date()
        default_split = close.index[int(n_bars * 0.65)].date()
        default_split = min(max(default_split, split_lo), split_hi)

        cs1, cs2, cs3 = st.columns(3)
        split_date = cs1.date_input(
            "Train / test split", value=default_split,
            min_value=split_lo, max_value=split_hi,
            help=f"Everything before this date tunes the parameters. Everything "
                 f"after is held back to check them. Bounds keep at least "
                 f"{MIN_TRAIN} bars on the train side and {MIN_TEST} on the test "
                 f"side.")
        objective = cs2.selectbox(
            "Objective", CAL.OBJECTIVES, index=0,
            format_func=lambda o: {"direction": "Direction accuracy",
                                   "spearman": "Rank correlation",
                                   "sharpe": "Sharpe ratio"}[o])
        score_horizon = cs3.number_input("Scoring horizon (bars)", 1, 60, 5, 1,
                                         help="How far ahead the direction is "
                                              "judged against.")

        cs4, cs5, cs6 = st.columns(3)
        n_trials = cs4.number_input("Search trials", 40, 2000, 250, 10,
                                    help="More trials search harder and overfit "
                                         "harder. The test column is what keeps "
                                         "that honest.")
        cost_bps = cs5.number_input("Cost per flip (bps)", 0.0, 50.0, 3.0, 0.5)
        run_null = cs6.checkbox("Also run the null model", value=True,
                                help="Repeats the whole search against a "
                                     "phase-randomised copy of your price series, "
                                     "to show what score chance reaches.")

        cs7, cs8 = st.columns(2)
        pool_bodies = cs7.multiselect(
            "Bodies the search may use", A.BODIES,
            default=["SUN", "MARS", "JUPITER", "SATURN", "URANUS", "NEPTUNE"],
            help="The Moon and Mercury move fast enough to form aspects "
                 "constantly, which buries the slower pairs.")
        pool_aspects = cs8.multiselect(
            "Aspects the search may use", A.ASPECTS_ALL,
            default=[0.0, 45.0, 90.0, 120.0, 135.0, 180.0],
            format_func=lambda a: f"{a:g} {A.aspect_label(a)}")

        cs9, cs10 = st.columns(2)
        min_trades = cs9.number_input("Minimum trades on train", 2, 200, 8, 1,
                                      help="Rejects candidates that take three "
                                           "trades in a decade and get lucky.")
        min_expo = cs10.slider("Minimum time in market", 0.0, 0.9, 0.15, 0.05)

        go_cal = st.button("Run calibration", type="primary", key="run_cal")

        if go_cal:
            if len(pool_bodies) < 2 or not pool_aspects:
                st.error("The search needs at least two bodies and one aspect.")
            else:
                try:
                    lons_s, ev_all = _precompute(
                        (first - pd.Timedelta(500, "D")).strftime("%Y-%m-%d"),
                        (last + pd.Timedelta(500, "D")).strftime("%Y-%m-%d"),
                        tuple(pool_bodies), tuple(pool_aspects), 8.0, frame)

                    bar = st.progress(0.0, text="Searching...")
                    res = CAL.search(
                        close, lons_s.index, ev_all, pd.Timestamp(split_date),
                        n_trials=int(n_trials), objective=objective,
                        horizon=int(score_horizon), tc_bps=float(cost_bps),
                        body_pool=pool_bodies, aspect_pool=pool_aspects,
                        min_trades=int(min_trades), min_exposure=float(min_expo),
                        progress=lambda i, n: bar.progress(
                            i / n, text=f"Trial {i} of {n}"),
                    )
                    bar.empty()

                    null = None
                    if run_null:
                        with st.spinner("Running the same search on randomised prices..."):
                            null = CAL.null_distribution(
                                close, lons_s.index, ev_all, pd.Timestamp(split_date),
                                n_trials=max(int(n_trials) // 3, 40),
                                objective=objective, horizon=int(score_horizon),
                                tc_bps=float(cost_bps), body_pool=pool_bodies,
                                aspect_pool=pool_aspects,
                                min_trades=int(min_trades),
                                min_exposure=float(min_expo))

                    st.session_state["cal_res"] = res
                    st.session_state["cal_null"] = null
                    st.session_state["cal_split"] = pd.Timestamp(split_date)
                    st.session_state["cal_obj"] = objective
                except Exception as exc:
                    st.error(f"Calibration failed: {exc}")

        res = st.session_state.get("cal_res")
        if res is None:
            st.info("Set the split date and press Run calibration.")
        else:
            null = st.session_state.get("cal_null")
            split_used = st.session_state.get("cal_split")
            obj_used = st.session_state.get("cal_obj", "direction")
            neutral = 0.5 if obj_used == "direction" else 0.0

            best = res.iloc[0]
            k1, k2, k3, k4, k5 = st.columns(5)
            kpi(k1, "Candidates", f"{len(res)}", f"split {split_used.date()}")
            kpi(k2, "Best on train", f"{best['train_score']:.4f}",
                "what tuning alone would pick")
            kpi(k3, "Its test score",
                f"{best['test_score']:.4f}" if pd.notna(best["test_score"]) else "n/a",
                "held-out data")
            kpi(k4, "Gap",
                f"{best['gap']:+.4f}" if pd.notna(best["gap"]) else "n/a",
                "train minus test")
            if null and pd.notna(null.get("median_test", np.nan)):
                kpi(k5, "Chance level", f"{null['median_test']:.4f}",
                    f"cherry-picked: {null['best_test']:.3f}")
            else:
                kpi(k5, "Neutral", f"{neutral:.2f}", "coin flip")

            # Verdict, stated as a comparison rather than an opinion.
            if null and pd.notna(null.get("median_test", np.nan)):
                edge = float(best["test_score"]) - float(null["median_test"])
                beat_cherry = float(best["test_score"]) > float(null["best_test"])
                if beat_cherry:
                    st.markdown(
                        f'<div class="note"><b>Test score {best["test_score"]:.4f} '
                        f'exceeds even the best score the same search reached on '
                        f'randomised prices ({null["best_test"]:.4f}).</b> That is '
                        f'the strongest result this tool can produce. Re-run with a '
                        f'different split date before acting on it: a single split '
                        f'can be lucky.</div>', unsafe_allow_html=True)
                elif edge > 0.02:
                    st.markdown(
                        f'<div class="warn"><b>Test score {best["test_score"]:.4f} '
                        f'is {edge:+.4f} above the chance median '
                        f'({null["median_test"]:.4f}), but below what cherry-picking '
                        f'on noise reaches ({null["best_test"]:.4f}).</b> Suggestive, '
                        f'not established. Try other split dates and see whether the '
                        f'same parameter family keeps winning.</div>',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<div class="warn"><b>Test score {best["test_score"]:.4f} '
                        f'sits at the chance median ({null["median_test"]:.4f}).</b> '
                        f'The search fitted the training window and nothing '
                        f'transferred. The train score is the search\'s flexibility, '
                        f'not a discovery. Changing bodies, aspects or the objective '
                        f'is worth trying; raising the trial count is not, and will '
                        f'only fit the training window harder.</div>',
                        unsafe_allow_html=True)

            st.markdown("#### Candidates, ranked by train score")
            st.caption(
                "Ranked by train because ranking by test and then reporting that "
                "test score is just overfitting to the test window instead. Look "
                "for a family of similar parameters that all hold up, rather than "
                "one standout row."
            )
            show_cols = ["train_score", "test_score", "gap", "test_sharpe",
                         "test_return", "test_trades", "exposure", "n_events", "label"]
            st.dataframe(res[show_cols].head(40), width="stretch", hide_index=False,
                         height=340)
            st.download_button("Download all candidates CSV",
                               res.to_csv(index=False).encode(),
                               file_name="kairos_calibration.csv", mime="text/csv")

            st.markdown("#### Apply a candidate")
            pick = bounded_number(st, "Candidate row number", 0,
                                  max(len(res) - 1, 0), 0, 1,
                                  help="Row numbers match the table above.")
            chosen = CAL.params_from_row(res.iloc[int(pick)])
            st.code(chosen.label(), language="text")

            # Equity on train and test, drawn separately so the join is visible.
            try:
                lons_a, ev_a = _precompute(
                    (first - pd.Timedelta(500, "D")).strftime("%Y-%m-%d"),
                    (last + pd.Timedelta(500, "D")).strftime("%Y-%m-%d"),
                    tuple(pool_bodies), tuple(pool_aspects), 8.0, frame)
                ev_c = CAL.filter_events(ev_a, chosen)
                wave_c = CAL.build_wave(lons_a.index, ev_c, chosen)
                det = CAL.signal_detail(wave_c, close.index, chosen)
                pos_c = det["position"]

                r1 = np.log(close).diff().fillna(0.0)
                traded = pos_c.shift(1).fillna(0.0)
                flips = traded.diff().abs().fillna(0.0)
                net = traded * r1 - flips * (float(cost_bps) / 1e4)
                eq = (1.0 + net).cumprod()
                bh = np.exp(np.log(close / close.iloc[0]))

                frame_eq = pd.DataFrame({
                    "strategy": eq / eq.iloc[0],
                    "buy and hold": bh,
                })
                st.plotly_chart(
                    C.multi_line_chart(frame_eq, title="Equity, whole range"),
                    width="stretch", theme=None)
                st.caption(
                    f"Vertical read: everything left of {split_used.date()} was "
                    f"used to choose these parameters, so of course it looks good. "
                    f"Only the part to the right is evidence."
                )

                tr_mask = close.index < split_used
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Train return", f"{(eq[tr_mask].iloc[-1] / eq.iloc[0] - 1) * 100:+.1f}%")
                m2.metric("Test return", f"{(eq.iloc[-1] / eq[tr_mask].iloc[-1] - 1) * 100:+.1f}%")
                m3.metric("Buy & hold, test",
                          f"{(bh.iloc[-1] / bh[tr_mask].iloc[-1] - 1) * 100:+.1f}%")
                m4.metric("Time in market", f"{float((pos_c != 0).mean()) * 100:.0f}%")
            except Exception as exc:
                st.warning(f"Could not rebuild the equity curve: {exc}")

            st.markdown("#### Projection past the last bar")
            fc1, fc2 = st.columns([1, 3])
            days_ahead = fc1.number_input("Days ahead", 30, 720, 180, 30)
            try:
                fwave, changes = CAL.forward_plan(
                    chosen, last, days_ahead=int(days_ahead), frame=frame)
                with fc2:
                    st.caption(
                        "The wave here is exact: it comes from orbital mechanics "
                        "and needs no market assumption. The positions derived "
                        "from it are only as good as the test score above."
                    )
                if changes.empty:
                    st.info("No position changes in this window.")
                else:
                    st.dataframe(changes, width="stretch", hide_index=True)
                    st.caption(
                        "`driven_by_date` is the date whose wave value produced the "
                        f"position, offset by {chosen.signal_offset:+d} days. Reading "
                        "the wave ahead is legitimate here, and is the one real edge "
                        "in this approach: planetary positions are computable years "
                        "in advance in a way no price indicator is. Future prices are "
                        "never read anywhere."
                    )
                    st.download_button("Download forward plan CSV",
                                       changes.to_csv(index=False).encode(),
                                       file_name="kairos_forward_plan.csv",
                                       mime="text/csv")

                drivers = CAL.upcoming_drivers(chosen, last, int(days_ahead), frame)
                if not drivers.empty:
                    st.markdown("##### Alignments inside the projection window")
                    st.dataframe(drivers, width="stretch", hide_index=True)
            except Exception as exc:
                st.error(f"Projection failed: {exc}")

# --------------------------------------------------------------------------
# Sky grid
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading tickers...", ttl=3600, max_entries=24)
def load_many(tickers: tuple, start: str, end: str, interval: str, demo: bool):
    out, errs = {}, {}
    for i, t in enumerate(tickers):
        try:
            if demo:
                out[t] = M.demo_ohlc(start, end, seed=7 + i * 13)["Close"]
            else:
                out[t] = M.fetch_ohlc(t, start=start, end=end,
                                      interval=interval)["Close"]
        except Exception as exc:
            errs[t] = str(exc)
    return out, errs


with tabs[0]:
    st.subheader("Gann sky grid")
    st.markdown(
        """
        **Vertical axis is degrees, 0 at the bottom to 180 at the top.** Each
        blue dot is the chosen planet's peak altitude for one interval, sitting
        at its measured angle with no conversion applied. Gann rays radiate
        from every dot in all four directions. The stock is the overlay,
        scaled into the degree band by a ratio you adjust until its past
        movement lines up with the planetary geometry. The red line is now.

        Working in degrees rather than dollars removes the per-instrument
        calibration from the rays entirely: the 1x1 is one degree per day,
        which is Gann's own unit and is identical on every instrument. Only
        the price overlay needs fitting.

        The crossings are the output. A ray cast backward from a **future**
        planet position can be crossed with a ray from a dot that has already
        happened, and that crossing is computable today because the future
        position is exact. Those are marked as `past-future` and are the
        decision candidates.
        """
    )

    # Interval first and on its own row, because it is the primary control
    # and it was previously buried among a dozen others.
    i1, i2, i3 = st.columns([1, 1, 2])
    dg_interval = i1.radio(
        "Dot interval", list(SG.INTERVALS.keys()), index=2, horizontal=True,
        key="dg_int",
        help="A dot for every one of these across the whole displayed "
             "timeline, past and future. Planetary positions are "
             "deterministic, so nothing is sampled or skipped.")
    dg_future = i2.number_input(
        "Future days beyond the window", 0, 3650, 180, 30, key="dg_ext",
        help="Independent of the Start and End dates in the sidebar. Those set "
             "how much history is shown; this sets how far past today the grid "
             "is projected.")
    i3.caption(
        "Start and End in the sidebar control the history window. The future "
        "extension above is separate and adds to the right of it, so you can "
        "show two years of past against six months of projection."
    )

    d1, d2, d3, d4 = st.columns(4)
    dg_ticks = d1.text_input("Tickers", value=ticker, key="dg_tick",
                             help="Comma separated. Each is scaled into the "
                                  "degree band separately.")
    dg_body = d2.multiselect("Planet of choice", A.BODIES, default=["MARS"],
                             key="dg_body",
                             help="Every planet selected gets its own dots and "
                                  "its own rays, in its own colour.")
    dg_lat = d3.number_input("Observer latitude", -66.0, 66.0, 47.8, 0.1,
                             key="dg_lat")
    dg_measure = d4.selectbox("Angle measure",
                              ["peak altitude (0-90)", "declination (-90..90)"],
                              index=0, key="dg_meas")

    d5, d6, d7, d8 = st.columns(4)
    dg_rate_mode = d5.radio(
        "1x1 rate", ["square the chart", "manual"], index=0, horizontal=True,
        key="dg_rmode",
        help="Squaring derives the rate from the window so a 1x1 ray always "
             "sits on the diagonal. A fixed rate cannot: 1 degree per day is "
             "a gentle slope across four months and near vertical across three "
             "years, which is what made every ray look like it went straight "
             "up and down.")
    dg_manual_rate = d6.number_input(
        "Manual rate (deg/day)", 0.005, 20.0, 0.5, 0.005, format="%.3f",
        key="dg_dpd", disabled=(dg_rate_mode == "square the chart"))
    dg_ratios = d7.multiselect("Ray ratios", list(G.FAN_RATIOS.keys()),
                               default=["1x2", "1x1", "2x1"], key="dg_ratios")
    dg_rayorigins = d8.slider(
        "Dots that emit rays", 1, 60, 8, 1, key="dg_origins",
        help="Every interval dot is always drawn. This limits only how many of "
             "them radiate rays, since rays are what fills the chart.")

    dr1, dr2 = st.columns(2)
    dg_raylen = dr1.slider("Ray length (days), 0 for unlimited",
                           0, 1000, 0, 10, key="dg_raylen",
                           help="Unlimited lets each ray run until it hits the "
                                "top or bottom of the degree band, which is "
                                "what the sketch shows. Set a value to cut them "
                                "short if the chart gets dense.")
    dg_dotsize = dr2.slider("Dot size", 4, 20, 9, 1, key="dg_dotsize")

    d9, d10, d11, d12 = st.columns(4)
    dg_originmode = d9.selectbox(
        "Ray origins chosen by",
        ["nearest now", "future only", "highest angle", "spread evenly"],
        index=0, key="dg_omode")
    dg_fwd = d10.checkbox("Forward rays", value=True, key="dg_fwd")
    dg_back = d11.checkbox("Backward rays", value=True, key="dg_back")
    dg_fast = d12.checkbox("Include fast ratios", value=False, key="dg_fast",
                           help="The steepest ratios cross the full 180 degrees "
                                "in days.")

    d13, d14 = st.columns(2)
    dg_autofit = d13.checkbox("Auto-fit the stock to the planet", value=True,
                              key="dg_autofit",
                              help="Least squares ratio and offset over the "
                                   "shared period. A starting position for the "
                                   "sliders, not evidence.")
    dg_marks = d14.checkbox("Mark confluence zones", value=False, key="dg_marks",
                            help="Off by default. These are the yellow crosses "
                                 "where several rays cross within a few days "
                                 "and degrees of each other. Useful in the "
                                 "table below; cluttered on the chart.")

    dg_tickers = tuple(x.strip().upper() for x in dg_ticks.split(",") if x.strip())
    dom = ((-90.0, 90.0) if dg_measure.startswith("declination")
           else (0.0, 180.0))

    if not dg_tickers:
        st.info("Enter at least one ticker.")
    elif not dg_body:
        st.info("Select a planet.")
    elif not dg_ratios:
        st.info("Select at least one ray ratio.")
    else:
        dg_px, dg_err = load_many(dg_tickers, start_s, end_s, interval, demo_mode)
        for tk, e in dg_err.items():
            st.warning(f"{tk}: {e}")

        manual_r, manual_o = {}, {}
        if not dg_autofit:
            st.markdown("###### Fit the stock into the degree band")
            fc = st.columns(min(len(dg_tickers), 4) * 2)
            for i, tk in enumerate(dg_tickers):
                manual_r[tk] = fc[(i * 2) % len(fc)].number_input(
                    f"{tk} ratio", 0.02, 3.0, 0.5, 0.02, key=f"dg_r_{tk}")
                manual_o[tk] = fc[(i * 2 + 1) % len(fc)].number_input(
                    f"{tk} offset (deg)", -90.0, 180.0, 10.0, 1.0,
                    key=f"dg_o_{tk}")

        if not dg_px:
            st.error("No ticker loaded. Check the symbols, or tick offline "
                     "demo data in the sidebar.")
        else:
            grid = SG.build_degree_grid(
                dg_px, bodies=tuple(dg_body), latitude=float(dg_lat),
                interval=dg_interval, domain=dom,
                use_declination=dg_measure.startswith("declination"),
                deg_per_day=(None if dg_rate_mode == "square the chart"
                             else float(dg_manual_rate)),
                ratios={k: G.FAN_RATIOS[k] for k in dg_ratios},
                stock_ratios=manual_r, stock_offsets=manual_o,
                autofit=bool(dg_autofit), extend_days=int(dg_future),
                max_ray_days=(int(dg_raylen) if int(dg_raylen) > 0 else None), forward=bool(dg_fwd),
                backward=bool(dg_back), include_fast=bool(dg_fast),
                ray_dots=int(dg_rayorigins),
                ray_dot_mode=dg_originmode,
            )
            if grid.get("error"):
                st.error(str(grid["error"]))
            else:
                gm = grid["meta"]
                q1, q2, q3, q4, q5 = st.columns(5)
                q1.metric("Dots", gm["n_dots"],
                          f"{gm['n_ray_origins']} emit rays")
                q2.metric("Rays", gm["n_rays"])
                q3.metric("Crossings", gm["n_intersections"])
                q4.metric("Confluences", gm["n_zones"])
                q5.metric("1x1", f"{gm['deg_per_day']:.3f} deg/day",
                          f"squared = {gm['squared_deg_per_day']:.3f}")

                st.plotly_chart(
                    C.degree_grid_chart(grid["curves"], grid["dots"],
                                        grid["shapes"], grid["now"],
                                        grid["x_range"], grid["domain"],
                                        grid["zones"] if dg_marks else None,
                                        dot_size=int(dg_dotsize)),
                    width="stretch", theme=None)

                if dg_autofit and grid["fits"]:
                    fitrows = []
                    for tk, f in grid["fits"].items():
                        fitrows.append({
                            "ticker": tk,
                            "ratio": round(f.get("ratio_used", np.nan), 4),
                            "offset_deg": round(f.get("offset_used", np.nan), 2),
                            "r_squared": (round(f["r_squared"], 4)
                                          if np.isfinite(f.get("r_squared", np.nan))
                                          else None),
                        })
                    st.dataframe(pd.DataFrame(fitrows), width="stretch",
                                 hide_index=True)
                    st.caption(
                        "r_squared measures how well the scaled price tracks "
                        "the planet's angle over the shared period. Two free "
                        "parameters on two smooth curves reach a respectable "
                        "figure with no relationship present, so use this to "
                        "position the overlay and use the Statistics tab to "
                        "ask whether it means anything."
                    )

                st.markdown("###### Decision candidates")
                st.caption(
                    "Crossings where one ray comes from a dot already behind "
                    "us and the other from a future dot. Both halves are exact "
                    "geometry; whether a crossing marks a turn is the empirical "
                    "question, and this table is what you would test."
                )
                inter = grid["intersections"]
                if inter.empty:
                    st.info("No crossings. Lengthen the rays or add a ratio.")
                else:
                    kinds = st.multiselect(
                        "Show crossing types",
                        ["past-future", "future-future", "past-past"],
                        default=["past-future"], key="dg_kinds")
                    ahead_only = st.checkbox("Only crossings ahead of now",
                                             value=True, key="dg_ahead")
                    view = inter[inter["kind"].isin(kinds)] if kinds else inter
                    if ahead_only:
                        view = view[view["days_from_now"] >= 0]
                    view = view.copy()
                    view["date"] = pd.to_datetime(view["date"]).dt.date
                    st.dataframe(view.sort_values("days_from_now"),
                                 width="stretch", hide_index=True, height=300)
                    st.download_button("Download crossings CSV",
                                       view.to_csv(index=False).encode(),
                                       file_name="kairos_crossings.csv",
                                       mime="text/csv")

                    zz = grid["zones"]
                    if not zz.empty:
                        st.markdown("###### Confluence zones, strongest first")
                        st.caption(
                            "Several crossings landing within a few days and "
                            "degrees of each other. One crossing of two lines "
                            "is weak; a cluster is the confluence the method "
                            "looks for. Counts compare only within this "
                            "configuration, since adding rays raises them all."
                        )
                        zv = zz.copy()
                        zv["date"] = pd.to_datetime(zv["date"]).dt.date
                        st.dataframe(zv.head(30), width="stretch",
                                     hide_index=True)

                with st.expander("Planet dots in degrees"):
                    dv = grid["dots"].copy()
                    dv["date"] = pd.to_datetime(dv["date"])
                    dv["future"] = dv["date"] > grid["now"]
                    dv["date"] = dv["date"].dt.date
                    dv["angle"] = dv["angle"].round(3)
                    st.dataframe(dv, width="stretch", hide_index=True)

# --------------------------------------------------------------------------
# Cycles
# --------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Cycles present in price, and synodic periods")
    st.caption(
        "Price is resampled onto calendar days before the spectrum is taken, "
        "because planetary periods are in calendar days and a period measured "
        "in trading days cannot be compared with them directly."
    )
    cc1, cc2, cc3 = st.columns(3)
    min_p = cc1.number_input("Shortest period (days)", value=25.0, min_value=4.0, step=5.0)
    max_p = cc2.number_input("Longest period (days)", value=2000.0, min_value=30.0, step=50.0)
    n_top = cc3.number_input("Peaks to report", value=10, min_value=1, max_value=40, step=1)
    tol = st.slider("Match tolerance (%)", 0.5, 20.0, 5.0, 0.5)

    spec = W.spectrum(close, min_period_days=float(min_p), max_period_days=float(max_p))
    peaks = W.dominant_cycles(close, min_period_days=float(min_p),
                              max_period_days=float(max_p), n_top=int(n_top))
    syn = load_synodics(tuple(bodies))

    st.plotly_chart(C.spectrum_chart(spec, peaks, syn.head(12)),
                    width="stretch", theme=None)

    if peaks.empty:
        st.info("Not enough history for a spectrum in this band.")
    else:
        matches = W.match_cycles_to_synodics(peaks, syn, tolerance_pct=float(tol))
        st.dataframe(matches, width="stretch", hide_index=True)
        n_hit = int(matches["within_tolerance"].sum()) if not matches.empty else 0
        st.markdown(
            f'<div class="warn"><b>{n_hit} of {len(matches)} detected cycles fall '
            f'within {tol:g}% of a synodic period or one of its first four '
            f'harmonics.</b> Before reading anything into that, note how easy the '
            f'match is: with {len(syn)} pairs and 4 harmonics each there are about '
            f'{len(syn) * 4} candidate periods scattered across the band, so a '
            f'close match is nearly guaranteed for any period at all. Running this '
            f'on the offline demo random walk typically "matches" most of its '
            f'cycles too. The columns that carry real information are '
            f'<code>miss_pct</code> and <code>cycles_in_sample</code>: a cycle '
            f'seen fewer than about five times in the data is not established, '
            f'however well it matches.</div>', unsafe_allow_html=True)

        st.markdown("#### Synodic periods for the selected bodies")
        st.dataframe(syn, width="stretch", hide_index=True)
        st.caption(
            "Measured from long-baseline mean motions rather than from your date "
            "range, so short windows do not corrupt slow pairs. Pairs involving "
            "the Moon are geocentric; the rest are heliocentric, because "
            "geocentric Sun-Mercury relative motion never accumulates."
        )

# --------------------------------------------------------------------------
# Calendar
# --------------------------------------------------------------------------
with tabs[6]:
    st.subheader("Alignment calendar")
    which = st.radio("Show", ["Upcoming", "Historical", "All"],
                     index=0, horizontal=True)
    if which == "Upcoming":
        table = events_future
    elif which == "Historical":
        table = events_hist
    else:
        table = events

    if table.empty:
        st.info("No events match the current aspect, orb and pair settings.")
    else:
        show = table.copy()
        show["date"] = pd.to_datetime(show["date"]).dt.date
        cols_order = ["date", "pair", "aspect_name", "aspect",
                      "separation", "offset", "weight"]
        show = show[[c for c in cols_order if c in show.columns]]
        show["separation"] = show["separation"].round(3)
        show["offset"] = show["offset"].round(3)
        show["weight"] = show["weight"].round(4)
        st.dataframe(show.sort_values("date"), width="stretch",
                     hide_index=True, height=420)
        st.download_button("Download alignments CSV",
                           data=show.to_csv(index=False).encode(),
                           file_name="kairos_alignments.csv", mime="text/csv")

    st.markdown("---")
    st.subheader("Stations and ingresses")
    st.caption(
        "Stations are dates a body changes between retrograde and direct "
        "motion; ingresses are crossings of a 30 degree sign boundary. Both are "
        "turning-point candidates in Gann's approach, independent of aspects."
    )
    stations = load_stations(frame_key(lons_full), lons_full)
    ing = A.ingress_events(lons_full)
    sc1, sc2 = st.columns(2)
    with sc1:
        st.caption(f"Stations ({len(stations)})")
        if not stations.empty:
            s2 = stations.copy()
            s2["date"] = pd.to_datetime(s2["date"]).dt.date
            s2["longitude"] = s2["longitude"].round(2)
            s2["zodiac"] = stations["longitude"].map(A.dms)
            st.dataframe(s2[["date", "body", "kind", "longitude", "zodiac"]],
                         width="stretch", hide_index=True, height=320)
    with sc2:
        st.caption(f"Ingresses ({len(ing)})")
        if not ing.empty:
            i2 = ing.copy()
            i2["date"] = pd.to_datetime(i2["date"]).dt.date
            i2["longitude"] = i2["longitude"].round(2)
            st.dataframe(i2[["date", "body", "sign", "longitude"]],
                         width="stretch", hide_index=True, height=320)

    st.markdown("---")
    st.subheader("Positions on a chosen date")
    pick = st.date_input("Date", value=min(date.today(), end_d),
                         min_value=date(1900, 1, 1), max_value=date(2100, 12, 31))
    snap = A.longitudes(pd.DatetimeIndex([pd.Timestamp(pick)]),
                        bodies=bodies, frame=frame)
    spd = A.daily_speed(A.longitudes(
        A.calendar_index(pd.Timestamp(pick) - pd.Timedelta(3, "D"),
                         pd.Timestamp(pick) + pd.Timedelta(3, "D")),
        bodies=bodies, frame=frame))
    snap_t = pd.DataFrame({
        "body": snap.columns,
        "longitude": [round(float(snap[c].iloc[0]), 4) for c in snap.columns],
        "zodiac": [A.dms(float(snap[c].iloc[0])) for c in snap.columns],
        "deg/day": [round(float(spd[c].iloc[3]), 4) if c in spd.columns else None
                    for c in snap.columns],
    })
    snap_t["motion"] = np.where(snap_t["deg/day"].fillna(0) < 0, "retrograde", "direct")
    st.dataframe(snap_t, width="stretch", hide_index=True)

# --------------------------------------------------------------------------
# Forecast models (optional heavy stack)
# --------------------------------------------------------------------------
with tabs[7]:
    st.subheader("Statistical and neural benchmark")

    # Availability is checked with find_spec rather than a try/except around
    # `import torch`, and the difference matters more than it looks.
    #
    # A half-installed torch - an interrupted download, a mismatched CUDA
    # runtime, a truncated .so from a full disk - does not raise ImportError.
    # It crashes the interpreter with SIGSEGV or SIGBUS, which no `except`
    # clause can catch, and it would take this entire app down with it while
    # showing the user nothing. Encountered exactly that during development
    # after a failed install, so it is not hypothetical.
    #
    # find_spec only looks for the module on disk; it never executes it. The
    # real import is deferred to the button handler below, where a crash is
    # at least attributable to an action the user just took.
    def ml_available() -> bool:
        try:
            import importlib.util
            return importlib.util.find_spec("torch") is not None
        except Exception:
            return False

    HAS_TORCH = ml_available()

    if not HAS_TORCH:
        st.markdown(
            '<div class="warn"><b>Machine-learning extras are not installed in '
            'this build.</b> They were left out on purpose: PyTorch, transformers '
            'and sentence-transformers add well over a gigabyte to a one-file '
            'executable and make it take a minute to start, for a benchmark most '
            'runs of this app never touch.<br><br>To enable them, run from source '
            'and install the extras:<br>'
            '<code>pip install -r requirements-ml.txt</code><br><br>'
            'Then rebuild with <code>BUILD_EXE.bat full</code> if you want them '
            'inside the executable as well.</div>', unsafe_allow_html=True)
        st.markdown(
            "The benchmark itself is intact in `stock_forecast/`: ARIMA, a "
            "conditional LSTM, a conditional WGAN-GP, LightGBM residual fusion "
            "and a meta-labelling filter, evaluated on walk-forward windows with "
            "Diebold-Mariano tests. It uses the alignment features from this app "
            "as conditioning inputs. Run it from the command line with:"
        )
        st.code("python scripts/run_benchmark.py", language="bash")
    else:
        st.caption(
            "Walk-forward benchmark using the alignment wave and harmonic "
            "indices as conditioning features. Training several models across "
            "rolling windows takes minutes, not seconds."
        )
        b1, b2, b3 = st.columns(3)
        seq_len = b1.number_input("Lookback (bars)", 20, 240, 60, 5)
        epochs = b2.number_input("Max epochs", 3, 300, 25, 1)
        n_windows = b3.number_input("Windows (0 = all)", 0, 20, 2, 1)
        go_ml = st.button("Run benchmark", type="secondary")
        if go_ml:
            log = st.empty()
            try:
                # First real import of torch in this process. If the install
                # is broken this is where it will fail, and the message below
                # is the last thing written before a hard crash would occur.
                log.write("Loading PyTorch...")
                import torch
                log.write(f"PyTorch {torch.__version__} on "
                          f"{'CUDA' if torch.cuda.is_available() else 'CPU'}")

                from stock_forecast.pipeline import run_benchmark_gui
                with st.spinner("Training..."):
                    res = run_benchmark_gui(
                        df=df, wave=wave_hist, lons=lons_hist,
                        seq_len=int(seq_len), epochs=int(epochs),
                        max_windows=int(n_windows), log=lambda m: log.write(m),
                    )
                st.success("Benchmark complete.")
                st.dataframe(res["summary"], width="stretch")
                st.caption(
                    "Read DPA against 0.5, which is what a coin flip scores. "
                    "The Persistence row is the honest floor: a model that "
                    "cannot beat it has learned nothing worth keeping, however "
                    "good its RMSE looks."
                )
                if res.get("predictions") is not None:
                    st.plotly_chart(
                        C.multi_line_chart(res["predictions"],
                                           title="First test window"),
                        width="stretch", theme=None)
            except Exception as exc:
                st.error(f"Benchmark failed: {exc}")
                st.code(traceback.format_exc())

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
with tabs[8]:
    st.subheader("Underlying data")
    d1, d2 = st.columns(2)
    with d1:
        st.caption("Price")
        st.dataframe(df.tail(400), width="stretch", height=340)
        st.download_button("Download price CSV", df.to_csv().encode(),
                           file_name="kairos_prices.csv", mime="text/csv")
    with d2:
        st.caption("Longitudes (degrees)")
        st.dataframe(lons_full.round(4).tail(400), width="stretch", height=340)
        st.download_button("Download longitudes CSV", lons_full.to_csv().encode(),
                           file_name="kairos_longitudes.csv", mime="text/csv")

    st.caption("Composite wave")
    wave_out = pd.DataFrame({
        "wave_z": wave_full.round(6),
        "is_projection": wave_full.index > last,
    })
    st.dataframe(wave_out.tail(400), width="stretch", height=280)
    st.download_button("Download wave CSV", wave_out.to_csv().encode(),
                       file_name="kairos_wave.csv", mime="text/csv")

    st.markdown("---")
    st.caption("Environment")
    import platform
    st.code(
        f"python        {platform.python_version()}\n"
        f"kairos        {__import__('kairos').__version__}\n"
        f"numpy         {np.__version__}\n"
        f"pandas        {pd.__version__}\n"
        f"streamlit     {st.__version__}\n"
        f"frozen        {P.is_frozen()}\n"
        f"app dir       {P.app_dir()}\n"
        f"bundle dir    {P.bundle_dir()}\n"
        f"cache dir     {P.cache_dir()}",
        language="text",
    )

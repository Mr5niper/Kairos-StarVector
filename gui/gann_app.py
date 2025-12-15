# gui/gann_app.py
import streamlit as st
import plotly.graph_objects as go
from stock_forecast.dataset import fetch_ohlc_yf
from stock_forecast.gann_grid import build_overlay_shapes, PLANETS
from datetime import date
st.set_page_config(page_title="Gann/Fan + Planetary Overlay", layout="wide")
with st.sidebar:
    ticker = st.text_input("Ticker", "SPY")
    start = st.date_input("Start", value=date(date.today().year, 1, 1),
                      min_value=date(1990,1,1), max_value=date(2100,1,1)).isoformat()
    end   = st.date_input("End", value=date.today(),
                      min_value=date(1990,1,1), max_value=date(2100,1,1)).isoformat()
    pA = st.selectbox("Planet A", PLANETS, index=2)
    pB = st.selectbox("Planet B", PLANETS, index=4)
    aspects = st.multiselect("Aspects", [0,30,45,60,90,120,135,144,150,180], default=[0,60,90,120,180])
    orb = st.slider("Orb (deg)", 0.5, 5.0, 2.0, 0.5)
    step = st.number_input("Horizontal step", value=72.0, step=1.0)
    extend = st.number_input("Extend days", value=120, step=10)
    anchors = st.number_input("Max anchors", 1, 200, 24, 1)
    scale = st.slider("1x1 slope scale", 0.10, 3.0, 1.0, 0.05)
    both = st.checkbox("Fans both directions", True)
    verts = st.checkbox("Verticals", True)
    run = st.button("Build")
st.title("Gann/Fan + Planetary Overlay")
if run:
    df = fetch_ohlc_yf(ticker, start=start, end=end).dropna()
    ratios_map = {"1x8":1/8,"1x4":1/4,"1x3":1/3,"1x2":1/2,"1x1":1,"2x1":2,"3x1":3,"4x1":4,"8x1":8}
    ratios = [ratios_map[r] for r in ["1x8","1x4","1x3","1x2","1x1","2x1","3x1","4x1","8x1"]] # Default ratios hardcoded in gann_app, assuming all used
    shapes = build_overlay_shapes(df.index, df["Close"], (pA,pB), aspects, orb,
                                  int(anchors), ratios,
                                  float(scale), int(extend), bool(both), bool(verts), float(step))
    end_ext = df.index[-1] + pd.Timedelta(days=int(extend))
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"]))
    fig.update_layout(xaxis=dict(range=[df.index[0], end_ext]), shapes=shapes, height=760, margin=dict(l=0,r=0,t=24,b=0))
    st.plotly_chart(fig, use_container_width=True)

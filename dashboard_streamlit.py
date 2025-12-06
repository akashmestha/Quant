# dashboard_streamlit.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import io
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# project modules
from analytics import (
    load_ticks,
    resample_ohlc,
    hedge_ratio_ols,
    spread_and_zscore,
    adf_test,
    rolling_correlation,
)
from database_utils import (
    save_uploaded_ohlc,
    get_uploaded_symbols,
    load_uploaded_ohlc,
)
from config import SYMBOLS, STREAMLIT_REFRESH_MS

# -------------------------
# Helpers
# -------------------------
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=True)
    return buf.getvalue().encode()

def df_to_csv_string(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.to_csv(buf, index=True)
    return buf.getvalue()

def safe_resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Wrapper to ensure resample_ohlc works and returns a DataFrame."""
    try:
        o = resample_ohlc(df, timeframe)
        return o
    except Exception as e:
        st.warning("Resample error: " + str(e))
        return pd.DataFrame()

# -------------------------
# Page config & auto-refresh
# -------------------------
st.set_page_config(page_title="Quant Dashboard", layout="wide", initial_sidebar_state="expanded")
# Auto-refresh to simulate near-real-time updates
st_autorefresh(interval=STREAMLIT_REFRESH_MS, limit=None, key="autorefresh")

st.title("📈 Quant Dashboard (Streamlit + Python)")

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Controls")
st.sidebar.subheader("OHLC File Upload (optional)")

uploaded_file = st.sidebar.file_uploader("Upload OHLC CSV", type=["csv"])
use_uploaded_data = uploaded_file is not None
uploaded_df = None

# fetch uploaded symbols saved in DB
try:
    uploaded_symbols = get_uploaded_symbols()
except Exception as e:
    # if DB has issues, fallback to empty list
    uploaded_symbols = []
    st.sidebar.error("Could not fetch uploaded symbols: " + str(e))

# build symbol list (live symbols + uploaded)
all_symbols = list(dict.fromkeys(SYMBOLS + uploaded_symbols))

st.sidebar.markdown("**Select assets**")
symbol_y = st.sidebar.selectbox("Y Symbol (Dependent)", all_symbols, index=0)
symbol_x = st.sidebar.selectbox("X Symbol (Hedge Asset)", all_symbols, index=0)

timeframe = st.sidebar.selectbox("Timeframe", ["1s", "1m", "5m"], index=0)
window = st.sidebar.slider("Rolling Window (for z-score/correlation)", min_value=20, max_value=500, value=60, step=10)

st.sidebar.markdown("---")
st.sidebar.subheader("Alerts")
enable_alerts = st.sidebar.checkbox("Enable Alerts", value=False)
alert_type = st.sidebar.selectbox("Alert Type", ["Z-score > threshold", "Z-score < threshold", "Spread > threshold", "Spread < threshold"])
alert_threshold = st.sidebar.number_input("Threshold", value=2.0, step=0.1)
st.sidebar.caption("*(Y = asset you're trading, X = hedge asset)*")

# -------------------------
# Handle uploaded CSV
# -------------------------
if use_uploaded_data:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
        uploaded_df.columns = [c.strip().lower() for c in uploaded_df.columns]
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        if not required.issubset(set(uploaded_df.columns)):
            st.sidebar.error("Uploaded CSV must include columns: timestamp, open, high, low, close, volume")
            use_uploaded_data = False
            uploaded_df = None
        else:
            uploaded_df["timestamp"] = pd.to_datetime(uploaded_df["timestamp"])
            uploaded_df = uploaded_df.rename(columns={"timestamp": "ts"})
            uploaded_df = uploaded_df.set_index("ts")
            st.sidebar.success("Uploaded CSV ready (not yet saved to DB).")
            st.sidebar.markdown("Label uploaded data to save to DB:")
            uploaded_label = st.sidebar.text_input("Symbol label (DB)", value=f"UPLOAD_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
            if st.sidebar.button("Save uploaded OHLC to Postgres"):
                # Save copies with index as ts column
                save_df = uploaded_df.reset_index().rename(columns={"ts": "ts"})
                # ensure ts column present and datetime
                save_df["ts"] = pd.to_datetime(save_df["ts"])
                try:
                    n = save_uploaded_ohlc(save_df, uploaded_label)
                    st.sidebar.success(f"Saved {n} rows to DB under symbol '{uploaded_label}'. Refreshing symbol list...")
                except Exception as e:
                    st.sidebar.error("Failed to save uploaded OHLC: " + str(e))
    except Exception as e:
        st.sidebar.error("Error parsing uploaded file: " + str(e))
        use_uploaded_data = False
        uploaded_df = None

# -------------------------
# Load source data (uploaded prioritized)
# -------------------------
def load_source_for_symbol(symbol: str, uploaded_df_local: pd.DataFrame):
    """
    Priority:
      1) If user is using uploaded_df in the current run -> return that
      2) If symbol exists in saved uploads DB -> load_uploaded_ohlc
      3) Else -> live ticks (load_ticks)
    """
    # 1) use temporary upload for both X and Y if present
    if uploaded_df_local is not None:
        # Return a frame with columns ts, open/high/low/close/volume (index is ts)
        return uploaded_df_local.reset_index().rename(columns={"ts": "ts"})
    # 2) check saved uploaded symbols
    if symbol in uploaded_symbols:
        try:
            df = load_uploaded_ohlc(symbol)
            if df is not None and not df.empty:
                # ensure dt index/column
                if "ts" in df.columns:
                    df["ts"] = pd.to_datetime(df["ts"])
                    return df.reset_index().rename(columns={"ts": "ts"})
                else:
                    df.index = pd.to_datetime(df.index)
                    df = df.reset_index().rename(columns={"index": "ts"})
                    return df
        except Exception as e:
            st.warning(f"Error loading uploaded OHLC for {symbol}: {e}")
    # 3) live ticks
    try:
        df_ticks = load_ticks(symbol)
    except Exception as e:
        st.error(f"Error loading ticks for {symbol}: {e}")
        return pd.DataFrame()

    if df_ticks is None or df_ticks.empty:
        return pd.DataFrame()

    # df_ticks returned has columns ts (datetime), price, qty
    # Ensure ts column exists
    if "ts" not in df_ticks.columns:
        df_ticks = df_ticks.reset_index().rename(columns={"index": "ts"})
    # return
    return df_ticks

# load for Y and X
dfy_raw = load_source_for_symbol(symbol_y, uploaded_df)
dfx_raw = load_source_for_symbol(symbol_x, uploaded_df)

# quick debug / status
with st.expander("Data preview & counts (debug)", expanded=False):
    st.write("DFY HEAD:")
    if dfy_raw is None or dfy_raw.empty:
        st.write("Empty")
    else:
        st.dataframe(dfy_raw.head())
        st.markdown(f"**DFY COUNT:** {len(dfy_raw)}")
    st.write("---")
    st.write("DFX HEAD:")
    if dfx_raw is None or dfx_raw.empty:
        st.write("Empty")
    else:
        st.dataframe(dfx_raw.head())
        st.markdown(f"**DFX COUNT:** {len(dfx_raw)}")

# If no data at all, stop
if dfy_raw is None or dfy_raw.empty:
    st.warning(f"No data available for {symbol_y}. Wait for ingestion or upload OHLC.")
    st.stop()

# -------------------------
# Resample to OHLC for plotting/analytics
# -------------------------
try:
    ohlc_y = safe_resample(dfy_raw, timeframe)
except Exception as e:
    st.error("Error resampling Y: " + str(e))
    st.stop()

try:
    ohlc_x = safe_resample(dfx_raw, timeframe)
except Exception as e:
    st.error("Error resampling X: " + str(e))
    st.stop()

if ohlc_y.empty:
    st.warning("Not enough Y data after resampling. Try a different timeframe.")
    st.stop()

# -------------------------
# Price chart (candlestick for Y)
# -------------------------
st.header("Price Chart")
st.subheader(f"{symbol_y} Price (Candlestick)")

fig_price = go.Figure()
fig_price.add_trace(
    go.Candlestick(
        x=ohlc_y.index,
        open=ohlc_y["open"],
        high=ohlc_y["high"],
        low=ohlc_y["low"],
        close=ohlc_y["close"],
        name="Price",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    )
)

# add volume as secondary axis if exists
if "volume" in ohlc_y.columns:
    fig_price.add_trace(
        go.Bar(
            x=ohlc_y.index,
            y=ohlc_y["volume"],
            name="Volume",
            marker_color="rgba(128,128,128,0.3)",
            yaxis="y2",
        )
    )

fig_price.update_layout(
    height=520,
    xaxis=dict(rangeslider=dict(visible=False), type="date"),
    yaxis=dict(domain=[0.30, 1.0], title="Price"),
    yaxis2=dict(domain=[0.0, 0.25], title="Volume"),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
fig_price.update_xaxes(rangemode="normal", autorange=True)
fig_price.update_yaxes(autorange=True)

st.plotly_chart(fig_price, use_container_width=True)

# -------------------------
# If same symbol chosen -> single-asset mode
# -------------------------
if symbol_y == symbol_x:
    st.info("Single-asset mode: pair analytics (hedge ratio / spread) require two different symbols.")
    st.stop()

# -------------------------
# Pair analytics
# -------------------------
st.header("Hedge Ratio & Statistical Analysis (Pair)")

# align close series
y_close = ohlc_y["close"].rename("y_close")
x_close = ohlc_x["close"].rename("x_close")

# ensure enough overlap
joint = pd.concat([y_close, x_close], axis=1).dropna()
if joint.shape[0] < 10:
    st.error("Not enough overlapping data points between the two symbols after resampling.")
    st.stop()

# Hedge ratio
hr = hedge_ratio_ols(joint["y_close"], joint["x_close"])
if hr is None:
    st.error("Could not compute hedge ratio (insufficient data).")
    st.stop()

beta = hr["beta"]
r2 = hr["r2"]

c1, c2 = st.columns(2)
c1.metric("Hedge Ratio (β)", f"{beta:.6f}")
c2.metric("R²", f"{r2:.4f}")

# Spread & z-score
sz = spread_and_zscore(joint["y_close"], joint["x_close"], hedge_beta=beta, window=window)
spread = sz["spread"]
zscore = sz["zscore"]

# Spread chart
st.subheader("Spread (Y - β * X)")
fig_spread = go.Figure()
fig_spread.add_trace(go.Scatter(x=spread.index, y=spread.values, mode="lines", name="Spread"))
fig_spread.update_layout(height=300, margin=dict(t=30, b=20))
st.plotly_chart(fig_spread, use_container_width=True)

# Z-score chart with ±2 lines
st.subheader("Z-Score")
fig_z = go.Figure()
fig_z.add_trace(go.Scatter(x=zscore.index, y=zscore.values, mode="lines", name="Z-Score"))
fig_z.add_hline(y=2, line=dict(color="red", dash="dash"))
fig_z.add_hline(y=-2, line=dict(color="green", dash="dash"))
fig_z.update_layout(height=300, margin=dict(t=30, b=20))
st.plotly_chart(fig_z, use_container_width=True)

# ADF test
st.subheader("ADF Test for Mean Reversion (Spread)")
try:
    adf_res = adf_test(spread)
    st.json(adf_res)
except Exception as e:
    st.write("ADF error:", e)

# Rolling correlation
st.subheader("Rolling Correlation")
corr = rolling_correlation(joint["y_close"], joint["x_close"], window=window)
fig_corr = go.Figure()
fig_corr.add_trace(go.Scatter(x=corr.index, y=corr.values, mode="lines", name="Rolling Corr"))
fig_corr.update_layout(height=300, margin=dict(t=30, b=20))
st.plotly_chart(fig_corr, use_container_width=True)

# -------------------------
# Alerts evaluation
# -------------------------
alert_triggered = False
alert_message = ""
if enable_alerts:
    latest_z = float(zscore.dropna().iloc[-1]) if not zscore.dropna().empty else 0.0
    latest_spread = float(spread.dropna().iloc[-1]) if not spread.dropna().empty else 0.0

    if alert_type == "Z-score > threshold" and latest_z > alert_threshold:
        alert_triggered = True
        alert_message = f"🚨 Z-score {latest_z:.2f} > {alert_threshold}"
    elif alert_type == "Z-score < threshold" and latest_z < alert_threshold:
        alert_triggered = True
        alert_message = f"🚨 Z-score {latest_z:.2f} < {alert_threshold}"
    elif alert_type == "Spread > threshold" and latest_spread > alert_threshold:
        alert_triggered = True
        alert_message = f"🚨 Spread {latest_spread:.4f} > {alert_threshold}"
    elif alert_type == "Spread < threshold" and latest_spread < alert_threshold:
        alert_triggered = True
        alert_message = f"🚨 Spread {latest_spread:.4f} < {alert_threshold}"

if enable_alerts:
    st.subheader("🔔 Alerts")
    if alert_triggered:
        st.error(alert_message)
    else:
        st.info("No alerts triggered.")

# -------------------------
# Export data
# -------------------------
st.header("📤 Export / Download")

colA, colB, colC = st.columns(3)
with colA:
    st.write("### OHLC (Y)")
    try:
        ohlc_csv = df_to_csv_string(ohlc_y.reset_index().rename(columns={"index": "ts"}))
        st.download_button("Download OHLC CSV (Y)", data=ohlc_csv, file_name=f"{symbol_y}_{timeframe}_ohlc.csv", mime="text/csv")
    except Exception as e:
        st.write("Export error:", e)

with colB:
    st.write("### Spread")
    spread_df = pd.DataFrame({"ts": spread.index, "spread": spread.values})
    st.download_button("Download Spread CSV", data=df_to_csv_string(spread_df), file_name=f"{symbol_y}_{symbol_x}_spread.csv", mime="text/csv")

with colC:
    st.write("### Z-Score")
    z_df = pd.DataFrame({"ts": zscore.index, "zscore": zscore.values})
    st.download_button("Download Z-Score CSV", data=df_to_csv_string(z_df), file_name=f"{symbol_y}_{symbol_x}_zscore.csv", mime="text/csv")

st.success("Dashboard rendered successfully (auto-refresh enabled).")

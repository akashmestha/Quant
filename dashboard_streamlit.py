# dashboard_streamlit.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import io
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
from backtest_pair import run_pairs_backtest


# project helper modules (keep these as in your project)
from chart_utils import update_candlestick, update_line
from chart_interactive import make_interactive_chart

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
# Session state defaults
# -------------------------
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = True

# Keep a small flag for UI (not persistence)
if "ui_theme" not in st.session_state:
    st.session_state.ui_theme = "dark"

# -------------------------
# Helpers
# -------------------------
def df_to_csv_string(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.to_csv(buf, index=True)
    return buf.getvalue()

def safe_resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    try:
        return resample_ohlc(df, timeframe)
    except Exception as e:
        st.warning("Resample error: " + str(e))
        return pd.DataFrame()

# -------------------------
# Page config & visual theme
# -------------------------
st.set_page_config(page_title="Quant Dashboard", layout="wide", initial_sidebar_state="expanded")

# -------------------------
# Custom CSS - TradingView Dark Theme
# -------------------------
st.markdown(
    """
    <style>
    /* Backgrounds & text */
    html, body, [class*="css"]  {
        background-color: #0d1117 !important;
        color: #e6edf3 !important;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0f1620 !important;
        border-right: 1px solid #22303a;
    }
    /* Titles */
    h1, h2, h3, h4 {
        color: #e6edf3 !important;
        font-weight: 600;
    }
    /* Metric boxes */
    div[data-testid="metric-container"] {
        background-color: #0f1620;
        border: 1px solid #22303a;
        border-radius: 8px;
        padding: 10px;
    }
    /* Buttons */
    button {
        border-radius: 6px !important;
    }
    .stDownloadButton>button {
        background-color: #26323a !important;
        color: #e6edf3 !important;
    }
    /* Small tweaks */
    .section-box {
        background:#0f1620;
        padding:12px;
        border-radius:8px;
        border:1px solid #22303a;
        margin-bottom:16px;
    }
    /* Remove background behind charts */
    .js-plotly-plot .plotly, .plot-container {
        background-color: transparent !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Auto-refresh logic (sidebar toggle controls this)
# -------------------------
if st.session_state.auto_refresh:
    st_autorefresh(interval=STREAMLIT_REFRESH_MS, limit=None, key="periodic_refresh")

# -------------------------
# Page title + top nav (visual only)
# -------------------------
st.title("📈 Quant Dashboard")
st.markdown(
    """
    <div style="
        background-color:#0f1620;
        padding:10px;
        border-radius:8px;
        margin-bottom:18px;
        border:1px solid #22303a;
    ">
        <span style="font-size:16px; margin-right:22px; color:#c9d1d9;">📉 Charts</span>
        <span style="font-size:16px; margin-right:22px; color:#8b949e;">📊 Analytics</span>
        <span style="font-size:16px; margin-right:22px; color:#8b949e;">🧪 Backtesting</span>
        <span style="font-size:16px; margin-right:22px; color:#8b949e;">⚙️ Settings</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Controls")
st.sidebar.markdown("---")
st.sidebar.subheader("Page Auto-Refresh")
auto_refresh_toggle = st.sidebar.checkbox("Enable Auto-Refresh", value=st.session_state.auto_refresh)
st.session_state.auto_refresh = auto_refresh_toggle

st.sidebar.subheader("OHLC File Upload")
uploaded_file = st.sidebar.file_uploader("Upload OHLC CSV", type=["csv"])
use_uploaded_data = uploaded_file is not None
uploaded_df = None

# fetch uploaded symbols saved in DB (safe)
try:
    uploaded_symbols = get_uploaded_symbols()
except Exception as e:
    uploaded_symbols = []
    st.sidebar.error("Could not fetch uploaded symbols: " + str(e))

# build symbol list (live symbols + uploaded)
all_symbols = list(dict.fromkeys(SYMBOLS + uploaded_symbols))

st.sidebar.markdown("**Select assets**")
symbol_y = st.sidebar.selectbox("Y Symbol (Dependent)", all_symbols, index=0)
symbol_x = st.sidebar.selectbox("X Symbol (Hedge Asset)", all_symbols, index=0)

timeframe = st.sidebar.selectbox("Timeframe", ["1s", "1m", "5m"], index=0)
window = st.sidebar.slider("Rolling Window (for z-score/correlation)", min_value=20, max_value=500, value=60, step=10)

st.sidebar.markdown("### 📊 Chart Settings")
chart_type = st.sidebar.selectbox("Chart Type", ["Candlestick", "OHLC", "Line", "Area"], index=0)
show_volume = st.sidebar.checkbox("Show Volume", value=True)

st.sidebar.markdown("### 📈 Overlays")
sma_lengths = st.sidebar.multiselect("SMA", [10, 20, 50, 100])
ema_lengths = st.sidebar.multiselect("EMA", [10, 20, 50, 100])
show_bbands = st.sidebar.checkbox("Bollinger Bands (20,2)", value=False)
show_vwap = st.sidebar.checkbox("VWAP", value=False)

st.sidebar.markdown("### 🛠 Chart Tools")
crosshair_mode = st.sidebar.selectbox("Crosshair Mode", ["Default", "Crosshair", "Zoom", "Pan"])
show_grid = st.sidebar.checkbox("Show Grid", value=True)
autoscale_y = st.sidebar.checkbox("Auto Scale Y-Axis", value=True)

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
                save_df = uploaded_df.reset_index().rename(columns={"ts": "ts"})
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
# Data source loader
# -------------------------
def load_source_for_symbol(symbol: str, uploaded_df_local: pd.DataFrame):
    if uploaded_df_local is not None:
        return uploaded_df_local.reset_index().rename(columns={"ts": "ts"})
    if symbol in uploaded_symbols:
        try:
            df = load_uploaded_ohlc(symbol)
            if df is not None and not df.empty:
                if "ts" in df.columns:
                    df["ts"] = pd.to_datetime(df["ts"])
                    return df.reset_index().rename(columns={"ts": "ts"})
                else:
                    df.index = pd.to_datetime(df.index)
                    df = df.reset_index().rename(columns={"index": "ts"})
                    return df
        except Exception as e:
            st.warning(f"Error loading uploaded OHLC for {symbol}: {e}")
    try:
        df_ticks = load_ticks(symbol)
    except Exception as e:
        st.error(f"Error loading ticks for {symbol}: {e}")
        return pd.DataFrame()
    if df_ticks is None or df_ticks.empty:
        return pd.DataFrame()
    if "ts" not in df_ticks.columns:
        df_ticks = df_ticks.reset_index().rename(columns={"index": "ts"})
    return df_ticks

# load for Y and X
dfy_raw = load_source_for_symbol(symbol_y, uploaded_df)
dfx_raw = load_source_for_symbol(symbol_x, uploaded_df)

# debug expander
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
# chart header (symbol, timeframe, last price)
last_price = ohlc_y["close"].iloc[-1] if "close" in ohlc_y.columns and not ohlc_y["close"].empty else float("nan")
st.markdown(
    f"""
    <div style='
        display:flex;
        justify-content:space-between;
        align-items:center;
        padding:8px 10px;
        background:#0f1620;
        border:1px solid #22303a;
        border-radius:8px;
        margin-bottom:8px;
    '>
        <div>
            <span style="font-size:18px; font-weight:600;">{symbol_y} Chart</span>
            <span style="color:#8b949e; margin-left:12px;">Timeframe: {timeframe}</span>
        </div>
        <div style="color:#26ff8a; font-weight:600;">
            Last Price: {last_price:.2f}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

fig_price = make_interactive_chart(
    ohlc=ohlc_y,
    chart_type=chart_type,
    show_volume=show_volume,
    sma_lengths=sma_lengths,
    ema_lengths=ema_lengths,
    show_bbands=show_bbands,
    show_vwap=show_vwap,
    crosshair_mode=crosshair_mode,
    show_grid=show_grid,
    autoscale_y=autoscale_y
)

# enforce PAN as default interaction mode
fig_price.update_layout(dragmode="pan")
st.plotly_chart(fig_price, use_container_width=True, key="price_chart")

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

y_close = ohlc_y["close"].rename("y_close")
x_close = ohlc_x["close"].rename("x_close")

joint = pd.concat([y_close, x_close], axis=1).dropna()
if joint.shape[0] < 10:
    st.error("Not enough overlapping data points between the two symbols after resampling.")
    st.stop()

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
fig_spread = update_line("spread_chart_state", spread, st.session_state)
fig_spread.update_layout(dragmode="pan")
st.plotly_chart(fig_spread, use_container_width=True, key="spread_chart")

# Z-score chart
st.subheader("Z-Score")
fig_z = update_line("zscore_chart_state", zscore, st.session_state)
fig_z.update_layout(dragmode="pan")
st.plotly_chart(fig_z, use_container_width=True, key="zscore_chart")

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
fig_corr = update_line("corr_chart_state", corr, st.session_state)
fig_corr.update_layout(dragmode="pan")
st.plotly_chart(fig_corr, use_container_width=True, key="corr_chart")


# -------------------------
# 🔙 Backtesting — Pairs Trading Strategy
# -------------------------

st.header("🔙 Backtesting — Pairs Trading Strategy")

entry_z_input = st.number_input("Entry Z-Score Threshold", value=2.0)
exit_z_input = st.number_input("Exit Z-Score Threshold", value=0.0)
position_size_input = st.number_input("Position Size ($)", value=1000)
stop_loss_input = st.number_input("Stop Loss (%)", value=10.0)

if st.button("Run Backtest"):
    summary, equity_curve, trades_df = run_pairs_backtest(
        ohlc_y,
        ohlc_x,
        beta=beta,
        entry_z=entry_z_input,
        exit_z=exit_z_input,
        position_size=position_size_input,
        window=window,
        stop_loss_pct=stop_loss_input
    )

    if summary is None:
        st.error("Not enough data to run backtest.")
    else:
        st.success("Backtest Completed!")

        # ---- Summary Cards ----
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total PnL", f"{summary['Total PnL']:.2f}")
        c2.metric("Max Drawdown", f"{summary['Max Drawdown']:.2f}")
        c3.metric("Sharpe Ratio", f"{summary['Sharpe Ratio']:.3f}")
        c4.metric("Trades", summary["Number of Trades"])

        # ---- Equity Curve ----
        st.subheader("📈 Equity Curve")
        st.line_chart(equity_curve)

        # ---- Trades Table ----
        st.subheader("📄 Trades Executed")
        st.dataframe(trades_df)


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
# Export / Download
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

st.success("Dashboard rendered successfully.")

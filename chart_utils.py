import plotly.graph_objects as go

# -----------------------------
# UTILITY: Build or update candlestick chart
# -----------------------------
def update_candlestick(identifier, ohlc, session_state):

    # Create state holder if not exists
    if identifier not in session_state:
        session_state[identifier] = {
            "fig": None,
            "last_ts": None
        }

    state = session_state[identifier]
    fig = state["fig"]

    # First build
    if fig is None:
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=ohlc.index,
                    open=ohlc['open'],
                    high=ohlc['high'],
                    low=ohlc['low'],
                    close=ohlc['close'],
                    name="Price"
                )
            ]
        )
        fig.update_layout(
            height=350, 
            xaxis_rangeslider_visible=False,
            margin=dict(l=20, r=20, t=25, b=20)
        )

        fig.update_layout(dragmode="pan")

        state["fig"] = fig
        state["last_ts"] = ohlc.index[-1]
        return fig

    # Incremental update when new candle appears
    latest_ts = ohlc.index[-1]

    if state["last_ts"] is None or latest_ts > state["last_ts"]:
        fig.data[0].x = list(fig.data[0].x) + [latest_ts]
        fig.data[0].open = list(fig.data[0].open) + [ohlc["open"][-1]]
        fig.data[0].high = list(fig.data[0].high) + [ohlc["high"][-1]]
        fig.data[0].low = list(fig.data[0].low) + [ohlc["low"][-1]]
        fig.data[0].close = list(fig.data[0].close) + [ohlc["close"][-1]]

        state["last_ts"] = latest_ts

    return fig


# -----------------------------
# UTILITY: Update line chart (Spread, Z-score, Correlation)
# -----------------------------
def update_line(identifier, series, session_state):

    if identifier not in session_state:
        session_state[identifier] = {
            "fig": None,
            "last_ts": None
        }

    state = session_state[identifier]
    fig = state["fig"]

    # First build
    if fig is None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines"))

        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=25, b=20)
        )
        fig.update_layout(dragmode="pan")

        state["fig"] = fig
        state["last_ts"] = series.index[-1]
        return fig

    latest_ts = series.index[-1]

    # Append only the last value
    if state["last_ts"] is None or latest_ts > state["last_ts"]:
        fig.data[0].x = list(fig.data[0].x) + [latest_ts]
        fig.data[0].y = list(fig.data[0].y) + [series.values[-1]]
        state["last_ts"] = latest_ts

    return fig
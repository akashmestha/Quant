import plotly.graph_objects as go
import pandas as pd
import numpy as np

def make_interactive_chart(ohlc, chart_type, show_volume, sma_lengths, ema_lengths,
                           show_bbands, show_vwap, crosshair_mode, show_grid,
                           autoscale_y):
    
    fig = go.Figure()

    # ------------------------------
    # MAIN PRICE CHART TYPE
    # ------------------------------
    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=ohlc.index,
            open=ohlc["open"],
            high=ohlc["high"],
            low=ohlc["low"],
            close=ohlc["close"],
            name="OHLC"
        ))

    elif chart_type == "OHLC":
        fig.add_trace(go.Ohlc(
            x=ohlc.index,
            open=ohlc["open"],
            high=ohlc["high"],
            low=ohlc["low"],
            close=ohlc["close"],
            name="OHLC"
        ))

    elif chart_type == "Line":
        fig.add_trace(go.Scatter(
            x=ohlc.index,
            y=ohlc["close"],
            mode="lines",
            name="Close"
        ))

    elif chart_type == "Area":
        fig.add_trace(go.Scatter(
            x=ohlc.index,
            y=ohlc["close"],
            mode="lines",
            fill="tozeroy",
            name="Close"
        ))

    # ------------------------------
    # OVERLAYS: SMA / EMA
    # ------------------------------
    for length in sma_lengths:
        fig.add_trace(go.Scatter(
            x=ohlc.index,
            y=ohlc["close"].rolling(length).mean(),
            mode="lines",
            name=f"SMA {length}"
        ))

    for length in ema_lengths:
        fig.add_trace(go.Scatter(
            x=ohlc.index,
            y=ohlc["close"].ewm(span=length).mean(),
            mode="lines",
            name=f"EMA {length}"
        ))

    # ------------------------------
    # Bollinger Bands (20, 2)
    # ------------------------------
    if show_bbands:
        sma20 = ohlc["close"].rolling(20).mean()
        std20 = ohlc["close"].rolling(20).std()

        upper = sma20 + 2 * std20
        lower = sma20 - 2 * std20

        fig.add_trace(go.Scatter(x=ohlc.index, y=upper, mode="lines", name="BB Upper"))
        fig.add_trace(go.Scatter(x=ohlc.index, y=lower, mode="lines", name="BB Lower"))

    # ------------------------------
    # VWAP (only if volume exists)
    # ------------------------------
    if show_vwap and ("volume" in ohlc.columns):
        tp = (ohlc["high"] + ohlc["low"] + ohlc["close"]) / 3
        vwap = (tp * ohlc["volume"]).cumsum() / ohlc["volume"].cumsum()

        fig.add_trace(go.Scatter(
            x=ohlc.index, y=vwap,
            mode="lines",
            name="VWAP"
        ))
    elif show_vwap:
        # no volume column found → skip VWAP
        pass
    

    # ------------------------------
    # Volume (secondary axis)
    # ------------------------------
    if show_volume and ("volume" in ohlc.columns):
        fig.add_trace(go.Bar(
            x=ohlc.index,
            y=ohlc["volume"],
            name="Volume",
            yaxis="y2",
            marker_color="rgba(150,150,150,0.3)"
        ))


    # ------------------------------
    # Chart Layout Settings
    # ------------------------------
    fig.update_layout(
        height=600,
        xaxis=dict(
            rangeslider=dict(visible=False),
            showgrid=show_grid
        ),
        yaxis=dict(
            showgrid=show_grid,
            autorange="reversed" if not autoscale_y else True
        ),
        yaxis2=dict(
            overlaying="y",
            side="right",
            showgrid=False,
            visible=show_volume
        ),
        dragmode=(
            "zoom" if crosshair_mode == "Zoom"
            else "pan" if crosshair_mode == "Pan"
            else "crosshair" if crosshair_mode == "Crosshair"
            else "zoom"
        ),
    )

    # Default pointer mode = PAN
    fig.update_layout(dragmode="pan")

    return fig

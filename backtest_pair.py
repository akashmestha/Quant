import pandas as pd
import numpy as np

def compute_zscore(spread, window):
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std()
    return (spread - mean) / std


def run_pairs_backtest(
    ohlc_y,
    ohlc_x,
    beta,
    entry_z=2.0,
    exit_z=0.0,
    position_size=1000,
    window=60,
    stop_loss_pct=10.0
):
    """
    Standard Pairs Trading Backtest
    Y = dependent asset
    X = hedge asset
    """
    df = pd.DataFrame()
    df["y"] = ohlc_y["close"]
    df["x"] = ohlc_x["close"]
    df["spread"] = df["y"] - beta * df["x"]
    df["zscore"] = compute_zscore(df["spread"], window)

    df = df.dropna()
    if df.empty:
        return None, None, None

    trades = []      # List of dicts
    equity = [0]     # Equity curve
    position = 0     # +1 long spread, -1 short spread
    entry_price = 0
    max_equity = 0

    for i in range(1, len(df)):
        z = df["zscore"].iloc[i]
        spread_now = df["spread"].iloc[i]
        spread_prev = df["spread"].iloc[i - 1]

        # ---------------------------
        # ENTRY CONDITIONS
        # ---------------------------
        if position == 0:
            # Short spread: Y overvalued
            if z > entry_z:
                position = -1
                entry_price = spread_now
                trades.append({
                    "type": "SHORT",
                    "entry_time": df.index[i],
                    "entry_spread": entry_price
                })

            # Long spread: Y undervalued
            elif z < -entry_z:
                position = 1
                entry_price = spread_now
                trades.append({
                    "type": "LONG",
                    "entry_time": df.index[i],
                    "entry_spread": entry_price
                })

        # ---------------------------
        # EXIT CONDITIONS
        # ---------------------------
        elif position != 0:
            # Exit when zscore returns to neutral
            if abs(z) < exit_z:
                pnl = (spread_now - entry_price) * position * position_size
                equity.append(equity[-1] + pnl)

                trades[-1]["exit_time"] = df.index[i]
                trades[-1]["exit_spread"] = spread_now
                trades[-1]["pnl"] = pnl

                position = 0

            # Stop loss
            elif abs((spread_now - entry_price) / entry_price) * 100 > stop_loss_pct:
                pnl = (spread_now - entry_price) * position * position_size
                equity.append(equity[-1] + pnl)

                trades[-1]["exit_time"] = df.index[i]
                trades[-1]["exit_spread"] = spread_now
                trades[-1]["pnl"] = pnl
                trades[-1]["stoploss"] = True

                position = 0

        # If no exit happened, continue equity flat
        if len(equity) == i:
            equity.append(equity[-1])

    equity_curve = pd.Series(equity, index=df.index[:len(equity)])

    # ---------------------------
    # PERFORMANCE SUMMARY
    # ---------------------------
    total_pnl = equity_curve.iloc[-1]
    max_dd = float((equity_curve.cummax() - equity_curve).max())

    returns = equity_curve.diff().fillna(0)
    if returns.std() == 0:
        sharpe = 0
    else:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)

    summary = {
        "Total PnL": total_pnl,
        "Max Drawdown": max_dd,
        "Sharpe Ratio": sharpe,
        "Number of Trades": len(trades)
    }

    return summary, equity_curve, pd.DataFrame(trades)

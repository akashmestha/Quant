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
    Robust Pairs Trading Backtest
    """

    # ------------------------------
    # Build merged dataframe
    # ------------------------------
    df = pd.DataFrame({
        "y": ohlc_y["close"],
        "x": ohlc_x["close"],
    }).dropna()

    # Align timestamps properly
    df = df.sort_index()

    # Compute spread
    df["spread"] = df["y"] - beta * df["x"]

    # Compute zscore
    df["zscore"] = compute_zscore(df["spread"], window)

    # Drop only rows that cannot compute zscore
    df = df[df["zscore"].notna()]

    # Stronger check
    if len(df) < window + 10:
        print(f"[BACKTEST ERROR] Not enough post-window data. Have {len(df)}, need {window+10}.")
        return None, None, None

    # ------------------------------
    # Initialize variables
    # ------------------------------
    trades = []
    equity = [0]
    position = 0
    entry_price = None
    max_equity = 0

    for i in range(1, len(df)):
        z = df["zscore"].iloc[i]
        spread_now = df["spread"].iloc[i]
        spread_prev = df["spread"].iloc[i - 1]

        # ------------------------------
        # ENTRY RULES
        # ------------------------------
        if position == 0:

            # SHORT the spread: Y overvalued vs X
            if z > entry_z:
                position = -1
                entry_price = spread_now
                trades.append({
                    "type": "SHORT",
                    "entry_time": df.index[i],
                    "entry_spread": entry_price
                })

            # LONG the spread: Y undervalued vs X
            elif z < -entry_z:
                position = 1
                entry_price = spread_now
                trades.append({
                    "type": "LONG",
                    "entry_time": df.index[i],
                    "entry_spread": entry_price
                })

        # ------------------------------
        # EXIT RULES
        # ------------------------------
        else:

            # Normal exit to mean
            if abs(z) < exit_z:

                pnl = (spread_now - entry_price) * position * position_size
                equity.append(equity[-1] + pnl)

                trades[-1]["exit_time"] = df.index[i]
                trades[-1]["exit_spread"] = spread_now
                trades[-1]["pnl"] = pnl

                position = 0

            # Stop-loss exit
            elif abs((spread_now - entry_price) / entry_price) * 100 > stop_loss_pct:

                pnl = (spread_now - entry_price) * position * position_size
                equity.append(equity[-1] + pnl)

                trades[-1]["exit_time"] = df.index[i]
                trades[-1]["exit_spread"] = spread_now
                trades[-1]["pnl"] = pnl
                trades[-1]["stoploss"] = True

                position = 0

        # Keep equity flat if no new PnL
        if len(equity) == i:
            equity.append(equity[-1])

    # Final equity curve indexed by df timestamps
    equity_curve = pd.Series(equity, index=df.index[:len(equity)])

    # ------------------------------
    # PERFORMANCE METRICS
    # ------------------------------
    total_pnl = equity_curve.iloc[-1]
    max_dd = float((equity_curve.cummax() - equity_curve).max())

    returns = equity_curve.diff().fillna(0)
    sharpe = 0 if returns.std() == 0 else (returns.mean() / returns.std()) * np.sqrt(252)

    summary = {
        "Total PnL": total_pnl,
        "Max Drawdown": max_dd,
        "Sharpe Ratio": sharpe,
        "Number of Trades": len(trades)
    }

    return summary, equity_curve, pd.DataFrame(trades)

# analytics.py
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
from models import SessionLocal, Tick


def load_ticks(symbol):
    db = SessionLocal()
    try:
        rows = (
            db.query(Tick.ts, Tick.price, Tick.qty)
            .filter(Tick.symbol == symbol)
            .order_by(Tick.ts.desc())
            .limit(50000)
            .all()
        )
        df = pd.DataFrame(rows, columns=["ts", "price", "qty"])
        df["ts"] = pd.to_datetime(df["ts"])
        return df.sort_values("ts")
    finally:
        db.close()


def resample_ohlc(df, tf):
    rule = tf.replace("s", "S").replace("m", "T")
    ohlc = df.resample(rule, on="ts").agg({
        "price": "ohlc",
        "qty": "sum"
    })
    ohlc.columns = ohlc.columns.map("_".join)
    ohlc.rename(columns={"price_open": "open",
                         "price_high": "high",
                         "price_low": "low",
                         "price_close": "close",
                         "qty_sum": "volume"}, inplace=True)
    return ohlc.dropna()


def hedge_ratio_ols(y, x):
    if len(y) < 20:
        return None
    model = LinearRegression()
    model.fit(x.values.reshape(-1, 1), y.values)
    beta = model.coef_[0]
    r2 = model.score(x.values.reshape(-1, 1), y.values)
    return {"beta": beta, "r2": r2}


def spread_and_zscore(y, x, hedge_beta, window=60):
    spread = y - hedge_beta * x
    z = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()
    return {"spread": spread, "zscore": z}


def adf_test(series):
    s = series.dropna()
    result = adfuller(s)
    return {
        "ADF Statistic": result[0],
        "p-value": result[1],
        "Used lags": result[2],
        "N obs": result[3]
    }


def rolling_correlation(y, x, window=60):
    return y.rolling(window).corr(x)

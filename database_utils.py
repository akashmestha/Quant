# database_utils.py
from models import SessionLocal, OHLCKline
import pandas as pd

def get_uploaded_symbols():
    db = SessionLocal()
    try:
        rows = db.query(OHLCKline.symbol).distinct().all()
        return [r[0] for r in rows]
    finally:
        db.close()


def load_uploaded_ohlc(symbol):
    db = SessionLocal()
    try:
        rows = (
            db.query(OHLCKline.ts, OHLCKline.open, OHLCKline.high,
                     OHLCKline.low, OHLCKline.close, OHLCKline.volume)
            .filter(OHLCKline.symbol == symbol)
            .order_by(OHLCKline.ts.asc())
            .all()
        )
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"])
        return df
    finally:
        db.close()


def save_uploaded_ohlc(df, symbol):
    db = SessionLocal()
    try:
        for _, row in df.iterrows():
            obj = OHLCKline(
                symbol=symbol,
                ts=row["ts"],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row.get("volume", 0)
            )
            db.add(obj)
        db.commit()
    finally:
        db.close()
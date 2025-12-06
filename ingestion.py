# ingestion.py
import threading, json, time, datetime
from websocket import create_connection
from models import SessionLocal, Tick
from config import SYMBOLS, BINANCE_WS


class BinanceIngestor(threading.Thread):
    def __init__(self, symbols=None):
        super().__init__(daemon=True)
        self.symbols = symbols or SYMBOLS
        self.running = True
        self.ws = None

    def _stream_url(self):
        streams = "/".join([s.lower() + "@trade" for s in self.symbols])
        return BINANCE_WS + streams

    def run(self):
        while self.running:
            try:
                url = self._stream_url()
                print("Ingestor: connecting to", url)
                self.ws = create_connection(url, timeout=20)
                print("Ingestor: connected.")

                while self.running:
                    msg = self.ws.recv()
                    if not msg:
                        continue

                    payload = json.loads(msg)
                    data = payload.get("data", payload)

                    price = float(data.get("p", 0))
                    qty = float(data.get("q", 0))
                    ts = int(data.get("T", time.time() * 1000))
                    ts_dt = datetime.datetime.utcfromtimestamp(ts / 1000)
                    symbol = data.get("s")

                    db = SessionLocal()
                    tick = Tick(ts=ts_dt, symbol=symbol, price=price, qty=qty)
                    db.add(tick)
                    db.commit()
                    db.close()

            except Exception as e:
                print("Ingestor error:", e)
                time.sleep(2)
            finally:
                try:
                    if self.ws:
                        self.ws.close()
                except:
                    pass

    def stop(self):
        self.running = False

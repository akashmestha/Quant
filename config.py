# config.py
# Central configuration for the project

DB_URI = "postgresql+psycopg2://gemuser:gem_pass_123@localhost:5432/gemscap"

# Binance websocket base (combined streams)
BINANCE_WS = "wss://stream.binance.com:9443/stream?streams="

# Default symbols to ingest
SYMBOLS = ["BTCUSDT", "ETHUSDT"]

# Streamlit auto-refresh interval (ms)
STREAMLIT_REFRESH_MS = 5000
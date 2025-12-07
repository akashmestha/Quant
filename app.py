# app.py
from flask import Flask, jsonify
from ingestion import BinanceIngestor
from models import init_db

app = Flask(__name__)

# Init DB
init_db()

# Start websocket ingestion
ingestor = BinanceIngestor()
ingestor.start()
print("✔ Binance Ingestor Started")

@app.route("/")
def home():
    return jsonify({"status": "running", "message": "Quant backend online"})


if __name__ == "__main__":
    print("🚀 Starting Flask API server...")
    app.run(host="0.0.0.0", port=5002)
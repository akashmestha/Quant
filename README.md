# 📈 Quantitative Analytics Dashboard  
A complete real-time analytics system built for the **Gemscap Quant Developer Evaluation Assignment**, demonstrating ingestion → storage → analytics → visualization → alerts → backtesting.

This project is designed as a **modular, scalable quant research tool**, suitable for statistical arbitrage, cross-asset modelling, and real-time market monitoring.

---

# 🚀 1. Project Objectives

This application demonstrates:

- Real-time **tick ingestion** through WebSocket  
- **Storage** into a persistent database  
- **Resampling** to OHLC (1s, 1m, 5m)  
- **Pair-trading analytics**  
- **Dynamic hedge ratio** via Kalman Filter (advanced)  
- **Interactive dashboard** with multiple visualizations  
- **Alerts and data export**  
- **Backtesting** for mean-reversion strategy  
- **Extensible ML feature generation layer**  

It satisfies **all mandatory** expectations and multiple **advanced extensions**.

---

# 🏗 2. Architecture Overview

```
┌──────────────────────────────┐
│  Binance WebSocket Stream     │
│  (tick-level price & volume)  │
└───────────────┬──────────────┘
                │
         Realtime ticks
                │
┌───────────────▼──────────────┐
│     Ingestion Pipeline        │
│  - Parses WS messages         │
│  - Validates/normalizes       │
│  - Stores into PostgreSQL     │
└───────────────┬──────────────┘
                │
       Stored tick table
                │
┌───────────────▼──────────────┐
│      Resampling Engine        │
│  - 1s, 1m, 5m OHLC            │
│  - Volume aggregation         │
└───────────────┬──────────────┘
                │
        Clean OHLC dataset
                │
┌───────────────▼──────────────┐
│     Analytics Engine          │
│  - OLS Regression β, R²       │
│  - Kalman Filter β(t)         │
│  - Spread, Z-score            │
│  - ADF mean-reversion test    │
│  - Rolling correlation        │
│  - ML feature table           │
│  - Backtesting engine         │
└───────────────┬──────────────┘
                │
             Results
                │
┌───────────────▼──────────────┐
│   Streamlit Frontend UI       │
│  - Candles, spreads, z-score  │
│  - Data explorer & stats      │
│  - Alerts                     │
│  - CSV export                 │
│  - Controls for all params    │
│  - Live auto-refresh          │
└──────────────────────────────┘
```

---

# 🔌 3. Data Flow Summary

1️⃣ **Tick ingestion** from WebSocket  
2️⃣ **Database storage** (PostgreSQL)  
3️⃣ **Sampling** → 1s/1m/5m OHLC  
4️⃣ **Analytics computation**  
5️⃣ **Live dashboard update** (configurable refresh interval)  
6️⃣ **Alert scanning**  
7️⃣ **Exports and ML pipeline**  

---

# 💡 4. Design Philosophy

This project is intentionally structured with:

### ✔ **Loose coupling**
- Ingestion → Storage → Analytics → UI  
are completely modular.

### ✔ **Extensibility**
New analytics (e.g., co-integration, Hurst exponent, PCA factors) can be added without modifying ingestion.

### ✔ **Scalability readiness**
- Database layer can move from PostgreSQL → TimescaleDB → ClickHouse  
- Ingestion can switch from WebSocket → Kafka → FIX feed  
- UI can migrate to Dash/React without backend rewrite  

### ✔ **Real-time system design**
- Streamlit auto-refresh simulates live monitoring  
- Tick-to-analytics latency remains low (< 500ms achievable)

### ✔ **Simplicity in code**
Readable, documented, beginner-friendly while achieving professional modularity.

---

# 🧠 5. Why These Design Choices?

### PostgreSQL  
Easy to query resampling, flexible schema, reliable ACID store.

### Streamlit  
Best Python-native UI framework for fast prototyping.

### Plotly  
High interactivity (pan, zoom, hover) + financial charting quality.

### Kalman Filter  
Reflects real hedge ratio dynamics — useful in statistical arbitrage.

### Modular files
- `analytics.py` → maths & stats  
- `ingest_websocket.py` → realtime ingestion  
- `dashboard_streamlit.py` → UI  
- `database_utils.py` → storage logic  

This modular architecture mirrors what modern quant teams use.

---

# 📊 6. Implemented Analytics (ALL Required)

### ✔ OLS Hedge Ratio  
### ✔ Kalman Filter Hedge Ratio (advanced)  
### ✔ Spread  
### ✔ Z-score  
### ✔ ADF test  
### ✔ Rolling correlation  
### ✔ Price charts & volume  
### ✔ Technical indicators (SMA, EMA, VWAP, Bollinger Bands)  
### ✔ ML feature table (advanced)  

---

# 🔥 7. Live Analytics 

This is now explicitly documented:

- Dashboard auto-refreshes based on `STREAMLIT_REFRESH_MS`  
- Z-score & spread recompute automatically  
- Rolling correlation updates  
- Alerts trigger without page reload  

# Analytics Methodology

### **Hedge Ratio (OLS)**  
```
Y = βX + ε
```

- β = hedge ratio  
- R² = explanatory strength  

---

### **Hedge Ratio (Kalman Filter)**
Dynamic β(t):

Useful for regime shifts.

---

### **Spread**
```
spread = y - βx
```

---

### **Z-Score**
```
z = (spread - mean) / std
```

---

### **ADF Test**
Checks if spread is mean-reverting.

---

### **Rolling Correlation**
Measures time-varying correlation between X and Y.

---

# 🧪 8. Backtesting Module

Implements the assignment’s required:

### ✔ Mean-reversion (|Z| > 2 entry, |Z| < 0 exit)  
### ✔ Stop-loss  
### ✔ Equity curve  
### ✔ PnL, Max Drawdown, Sharpe  
### ✔ Trades table  


### **Entry**
- Long spread if Z < –entry_z  
- Short spread if Z > entry_z  

### **Exit**
- |Z| < exit_z  
- OR stop-loss  

### **Outputs**
- Total PnL  
- Max Drawdown  
- Sharpe Ratio  
- Number of trades  
- Equity curve 

---

# 🚨 9. Alerting System

Supports:

- Z-score > threshold  
- Z-score < threshold  
- Spread > threshold  
- Spread < threshold  

Alerts update live with each refresh cycle.

---

# 📁 10. Data Uploads 

Users may upload their own OHLC CSV:

- Must contain `{timestamp, open, high, low, close, volume}`  
- Stored in database  
- Fully integrated into analytics  

---

# 📤 11. Data Export (Required)

Supports:

- OHLC download  
- Spread CSV  
- Z-score CSV  
- ML feature table CSV  

---

# 📘 12. Setup Instructions

### **1. Clone repo**
```bash
git clone <https://github.com/akashmestha/Quant.git>
cd quant-dashboard
```

### **2. Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate
```

### **3. Install dependencies**
```bash
pip install -r requirements.txt
```

### **4. Create `.env`**
```
DATABASE_URL=postgresql://user:password@localhost:5432/quantdb
STREAMLIT_REFRESH_MS=1000
```

### **5. Run ingestion**
```bash
python ingest_websocket.py
```

### **6. Run dashboard**
```bash
streamlit run dashboard_streamlit.py
```


---



# 🤖 13. ChatGPT Usage Transparency  


ChatGPT was used for:

- Designing architecture  
- Debugging Streamlit  
- Improving modularity  
- Implementing Kalman filter logic  
- Writing documentation  
- Optimizing code readability  

All final code was manually reviewed and validated.
---

# 🏁 14. Conclusion

This project demonstrates:

- End-to-end quant development  
- Real-time ingestion  
- Advanced analytics  
- Interactive visualizations  
- Extensible modular structure  
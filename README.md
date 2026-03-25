# StockSense — NSE/BSE ML Predictor
**Student Project | Priyanshu Kumar | Roll: 23053289**

Random Forest model that predicts next-day UP/DOWN movement for NSE/BSE stocks.

---

## Project Structure
```
stock-predictor/
├── backend/
│   ├── app.py              # Flask API + Random Forest model
│   ├── requirements.txt    # Python dependencies
│   └── render.yaml         # Render.com deployment config
└── frontend/
    ├── index.html          # Full UI (single file, no build needed)
    └── vercel.json         # Vercel deployment config
```

---

## How It Works
1. Fetches 2 years of OHLCV data from Yahoo Finance (`yfinance`)
2. Computes 14 technical indicators: RSI, MACD, Bollinger Bands, ATR, OBV, SMA, momentum, etc.
3. Labels each day: next close > today close → **1 (UP)**, else **0 (DOWN)**
4. Trains a Random Forest (200 trees) on historical data
5. Predicts the direction for the next trading day

---

## Local Development

### Backend (Python)
```bash
cd backend
pip install -r requirements.txt
python app.py
# Runs on http://localhost:5000
```

Test it:
```bash
curl "http://localhost:5000/predict?ticker=RELIANCE"
```

### Frontend
Open `frontend/index.html` in your browser.
Change `API_BASE` in the script to `http://localhost:5000` for local testing.

---

## Free Deployment (Student-Friendly)

### Step 1 — Deploy Backend on Render.com (FREE)
1. Go to https://render.com → Sign up with GitHub (free)
2. Click **New → Web Service**
3. Connect your GitHub repo
4. Set **Root Directory** → `backend`
5. Set **Build Command** → `pip install -r requirements.txt`
6. Set **Start Command** → `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
7. Select **Free plan** → Deploy
8. Copy your app URL: `https://YOUR-APP-NAME.onrender.com`

> Note: Free tier sleeps after 15 mins of inactivity. First request may take ~30 seconds to wake up.

### Step 2 — Update Frontend
In `frontend/index.html`, find this line:
```javascript
const API_BASE = "https://YOUR-APP-NAME.onrender.com";
```
Replace with your actual Render URL.

### Step 3 — Deploy Frontend on Vercel (FREE)
1. Go to https://vercel.com → Sign up with GitHub (free)
2. Click **New Project** → Import your GitHub repo
3. Set **Root Directory** → `frontend`
4. Click **Deploy** — done!
5. Your app is live at `https://YOUR-PROJECT.vercel.app`

---

## Features
| Feature | Description |
|---------|-------------|
| RSI (14) | Relative Strength Index — overbought/oversold |
| MACD | Trend-following momentum indicator |
| Bollinger Bands | Volatility + price position |
| SMA 20/50 ratio | Short vs long-term trend |
| ATR % | Normalized volatility |
| Volume ratio | Volume vs 20-day average |
| OBV change | On-balance volume momentum |
| Price momentum 5/10d | Recent price momentum |

---

## Tech Stack
- **Backend**: Python, Flask, scikit-learn, yfinance, pandas, numpy
- **Frontend**: HTML, CSS, Vanilla JS (no framework needed)
- **Deployment**: Render.com (backend) + Vercel (frontend) — both FREE

---

*Disclaimer: Academic project only. Not financial advice.*

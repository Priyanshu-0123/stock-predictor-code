from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

def compute_features(df):
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # Returns
    df["daily_return"] = close.pct_change()

    # Moving averages
    df["sma_20"] = close.rolling(20).mean()
    df["sma_50"] = close.rolling(50).mean()
    df["sma_ratio"] = df["sma_20"] / df["sma_50"]

    # RSI (14)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma20
    df["bb_position"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)

    # ATR (volatility)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr"] / close

    # Volume
    df["volume_change"] = volume.pct_change()
    df["volume_sma"] = volume.rolling(20).mean()
    df["volume_ratio"] = volume / (df["volume_sma"] + 1e-9)

    # OBV
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    df["obv_change"] = obv.pct_change()

    # Price momentum
    df["momentum_5"] = close.pct_change(5)
    df["momentum_10"] = close.pct_change(10)

    # Target: next day up (1) or down (0)
    df["target"] = (close.shift(-1) > close).astype(int)

    return df

FEATURE_COLS = [
    "daily_return", "sma_ratio", "rsi", "macd", "macd_signal", "macd_hist",
    "bb_width", "bb_position", "atr_pct", "volume_change", "volume_ratio",
    "obv_change", "momentum_5", "momentum_10"
]

def train_and_predict(ticker):
    # Append .NS for NSE stocks if not already present
    if not ticker.endswith(".NS") and not ticker.endswith(".BO"):
        ticker = ticker.upper() + ".NS"

    df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
    if df.empty or len(df) < 100:
        return None, "Not enough data for this ticker. Try a valid NSE symbol like RELIANCE, TCS, INFY."

    df = compute_features(df)
    df = df.dropna()

    if len(df) < 60:
        return None, "Insufficient clean data after feature computation."

    X = df[FEATURE_COLS].values
    y = df["target"].values

    # Train on all but last row; predict last row
    X_train, X_test = X[:-1], X[-1:]
    y_train = y[:-1]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X_train_s, y_train)

    prob = model.predict_proba(X_test_s)[0]
    pred = int(model.predict(X_test_s)[0])

    # Feature importances
    importances = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))
    top_features = sorted(importances.items(), key=lambda x: -x[1])[:5]

    # Last row stats
    last = df.iloc[-1]
    current_price = float(df["Close"].iloc[-1])
    prev_price = float(df["Close"].iloc[-2])
    price_change_pct = ((current_price - prev_price) / prev_price) * 100

    # In-sample accuracy (last 30 days)
    recent_preds = model.predict(scaler.transform(X[-31:-1]))
    recent_actual = y[-31:-1]
    accuracy = float(np.mean(recent_preds == recent_actual)) * 100

    return {
        "ticker": ticker,
        "prediction": "UP" if pred == 1 else "DOWN",
        "confidence": round(float(max(prob)) * 100, 1),
        "prob_up": round(float(prob[1]) * 100, 1),
        "prob_down": round(float(prob[0]) * 100, 1),
        "current_price": round(current_price, 2),
        "price_change_pct": round(price_change_pct, 2),
        "rsi": round(float(last["rsi"]), 1),
        "macd": round(float(last["macd"]), 3),
        "bb_position": round(float(last["bb_position"]) * 100, 1),
        "atr_pct": round(float(last["atr_pct"]) * 100, 2),
        "recent_accuracy": round(accuracy, 1),
        "top_features": [{"name": k, "importance": round(v * 100, 1)} for k, v in top_features],
        "data_points": len(df),
    }, None

@app.route("/predict", methods=["GET"])
def predict():
    ticker = request.args.get("ticker", "").strip()
    if not ticker:
        return jsonify({"error": "Please provide a ticker symbol."}), 400

    result, error = train_and_predict(ticker)
    if error:
        return jsonify({"error": error}), 400

    return jsonify(result)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)

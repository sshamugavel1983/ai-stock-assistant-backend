import os
import yfinance as yf
import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from sklearn.preprocessing import MinMaxScaler

app = FastAPI()

def load_model_and_scaler(ticker):
    """Load trained LSTM model and scaler for a specific stock"""
    model_path = f"models/{ticker}_lstm.h5"
    scaler_path = f"models/{ticker}_scaler.npy"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None

    model = tf.keras.models.load_model(model_path)
    scaler = np.load(scaler_path, allow_pickle=True).item()
    return model, scaler

def get_recent_stock_data(ticker):
    """Fetch last 10 days of stock data for prediction using Yahoo Finance"""
    stock = yf.Ticker(ticker)
    df = stock.history(period="15d")["Close"]

    if df.empty:
        return None

    return df.values[-10:].reshape(-1, 1)  # Get last 10 days

@app.get("/predict")
def predict_stock_price(ticker: str):
    """Predict next day's stock price for any stock"""
    model, scaler = load_model_and_scaler(ticker)

    if model is None:
        return {"error": f"Model for {ticker} not found. Please train the model first."}

    recent_prices = get_recent_stock_data(ticker)
    if recent_prices is None:
        return {"error": f"Stock data not available for {ticker}"}

    # Normalize & reshape for LSTM input
    recent_prices_scaled = scaler.transform(recent_prices)
    X_input = np.array([recent_prices_scaled]).reshape(1, 10, 1)

    # Predict next day's stock price
    predicted_price_scaled = model.predict(X_input)
    predicted_price = scaler.inverse_transform(predicted_price_scaled)[0][0]

    return {"ticker": ticker, "predicted_price": float(predicted_price)}

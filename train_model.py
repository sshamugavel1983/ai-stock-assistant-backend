import os
import yfinance as yf
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Ensure models directory exists
os.makedirs("models", exist_ok=True)


def get_stock_data(ticker):
    """Fetch historical stock data from Yahoo Finance"""
    stock = yf.Ticker(ticker)
    df = stock.history(period="2y")["Close"]

    if df.empty:
        raise ValueError(f"No data found for {ticker}")

    return df


def create_sequences(data, seq_length=10):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def train_lstm(ticker):
    """Train LSTM model for a specific stock"""
    df = get_stock_data(ticker)

    # Normalize data
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df.values.reshape(-1, 1))

    # Create sequences
    X, y = create_sequences(data)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # Reshape for LSTM

    # Build LSTM model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, y, batch_size=16, epochs=10, verbose=1)

    # Save model and scaler
    model.save(f"models/{ticker}_lstm.h5")
    np.save(f"models/{ticker}_scaler.npy", scaler)

    print(f"Model trained & saved for {ticker}")


# Example: Train for multiple stocks
for stock in ["AAPL", "MSFT", "GOOGL", "AMZN"]:
    train_lstm(stock)

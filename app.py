# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import io
import tempfile
import datetime

st.set_page_config(page_title="Stock Price Prediction (LSTM)", layout="wide")

st.title("Stock Price Prediction — LSTM")
st.write("Train a simple LSTM on close prices and visualize predictions.")

# ---------------------------
# Try to import TensorFlow (optional)
# ---------------------------
tf = None
try:
    import tensorflow as tf  # noqa: E402
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dropout, Dense
    TF_AVAILABLE = True
except Exception as e:
    TF_AVAILABLE = False
    # keep names for type checking later
    Sequential = None
    load_model = None
    LSTM = None
    Dropout = None
    Dense = None

# ---------------------------
# Sidebar / user controls
# ---------------------------
st.sidebar.header("Controls")

ticker = st.sidebar.text_input("Ticker (yfinance)", value="GOOG").upper()
today = datetime.date.today()
start_date = st.sidebar.date_input("Start date", value=datetime.date(2012, 1, 1))
end_date = st.sidebar.date_input("End date", value=today)

if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")

train_split = st.sidebar.slider("Train proportion", min_value=0.5, max_value=0.95, value=0.80, step=0.01)
lookback = st.sidebar.number_input("Lookback (days)", min_value=10, max_value=365, value=100, step=10)
epochs = st.sidebar.slider("Epochs", min_value=1, max_value=200, value=50)
batch_size = st.sidebar.selectbox("Batch size", [16, 32, 64, 128], index=1)
random_seed = st.sidebar.number_input("Random seed", value=42)

download_model_name = st.sidebar.text_input("Saved model filename", value="stock_model.keras")

st.sidebar.markdown("---")
if not TF_AVAILABLE:
    st.sidebar.warning("TensorFlow NOT available in this env. Training will be disabled. Install TF or use a conda env to enable training.")

st.sidebar.write("- Training an LSTM can be slow locally. Reduce epochs/lookback for testing.")

# ---------------------------
# Helpers
# ---------------------------
@st.cache_data(ttl=3600)
def load_stock(ticker: str, start_date, end_date):
    """
    Safe download:
    - clamp end_date to today
    - ensure start < end
    - return df with Date, Close or None if empty
    """
    try:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
    except Exception:
        return None

    today_pd = pd.Timestamp.today().normalize()
    if end > today_pd:
        end = today_pd

    if start >= end:
        # fallback: use 1 year before end
        start = end - pd.Timedelta(days=365)

    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    try:
        df = yf.download(ticker, start=start_str, end=end_str, progress=False)
    except Exception:
        return None

    if df is None or df.empty:
        return None

    df = df.reset_index()
    if 'Close' not in df.columns:
        return None

    df = df[['Date', 'Close']].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def build_model(input_shape, seed=42):
    tf.random.set_seed(seed)
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(60, activation='relu', return_sequences=True),
        Dropout(0.3),
        LSTM(80, activation='relu', return_sequences=True),
        Dropout(0.4),
        LSTM(120, activation='relu'),
        Dropout(0.5),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def prepare_data(series: np.ndarray, lookback: int, train_prop: float):
    N = len(series)
    train_n = int(N * train_prop)
    train = series[:train_n]
    test = series[train_n - lookback:]  # include last lookback days of train for continuity

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train.reshape(-1, 1))
    test_scaled = scaler.transform(test.reshape(-1, 1))

    def gen_xy(arr):
        X, y = [], []
        for i in range(lookback, len(arr)):
            X.append(arr[i - lookback:i, 0])
            y.append(arr[i, 0])
        X = np.array(X)
        y = np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        return X, y

    X_train, y_train = gen_xy(train_scaled)
    X_test, y_test = gen_xy(test_scaled)

    return X_train, y_train, X_test, y_test, scaler


def inverse_scale(y_scaled, scaler):
    scale = 1.0 / scaler.scale_[0]
    return y_scaled * scale


def metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mean_true = np.mean(y_true) if np.mean(y_true) != 0 else 1.0
    return {
        "MSE": mse,
        "RMSE": rmse,
        "RMSE %": (rmse / mean_true) * 100,
        "MAE": mae,
        "MAE %": (mae / mean_true) * 100
    }


# ---------------------------
# Load data and show overview
# ---------------------------
df = load_stock(ticker, start_date, end_date)

if df is None or df.empty:
    st.error("No data found for ticker or invalid date range. Try another ticker.")
    st.stop()

st.subheader(f"{ticker} — Data from {df['Date'].min().date()} to {df['Date'].max().date()}")

# Convert min/max dates to clean strings
min_date_str = pd.to_datetime(df['Date'].min()).strftime("%Y-%m-%d")
max_date_str = pd.to_datetime(df['Date'].max()).strftime("%Y-%m-%d")

col1, col2 = st.columns((3, 1))

with col1:
    st.line_chart(df.set_index('Date')['Close'], height=350, use_container_width=True)

with col2:
    st.metric("Total rows", len(df))
    st.metric("Start date", min_date_str)
    st.metric("End date", max_date_str)

# Moving averages
ma1 = df['Close'].rolling(window=lookback).mean()
ma2 = df['Close'].rolling(window=2 * lookback).mean()

st.write("Moving averages (red = lookback, blue = 2× lookback)")

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(df['Date'], df['Close'], label='Close', linewidth=1)
ax.plot(df['Date'], ma1, 'r', linewidth=1, label=f'MA {lookback}')
ax.plot(df['Date'], ma2, 'b', linewidth=1, label=f'MA {2 * lookback}')
ax.legend()
st.pyplot(fig)

# ---------------------------
# Train / Predict
# ---------------------------
train_button = st.button("Train model (this may take a while)")

if train_button:
    if not TF_AVAILABLE:
        st.error("TensorFlow is not available in this environment. Training is disabled.")
    else:
        seed = int(random_seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        st.info("Preparing data...")
        prices = df['Close'].values
        X_train, y_train, X_test, y_test, scaler = prepare_data(prices, int(lookback), float(train_split))

        st.write(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

        model = build_model((X_train.shape[1], 1), seed=seed)

        # train with spinner
        with st.spinner("Training model... This will block the UI until finished."):
            history = model.fit(X_train, y_train, epochs=int(epochs), batch_size=int(batch_size), verbose=1)

        st.success("Training finished")

        # Predict & inverse scale
        y_pred_scaled = model.predict(X_test)
        y_pred = inverse_scale(y_pred_scaled.flatten(), scaler)
        y_true = inverse_scale(y_test.flatten(), scaler)

        m = metrics(y_true, y_pred)
        st.subheader("Metrics")
        st.write(f"MSE: {m['MSE']:.4f}")
        st.write(f"RMSE: {m['RMSE']:.4f} ({m['RMSE %']:.2f}% of mean actual)")
        st.write(f"MAE: {m['MAE']:.4f} ({m['MAE %']:.2f}% of mean actual)")

        # Plot actual vs predicted
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ax2.plot(y_true, label='Actual', linewidth=1)
        ax2.plot(y_pred, label='Predicted', linewidth=1)
        ax2.set_title("Actual vs Predicted (test set)")
        ax2.legend()
        st.pyplot(fig2)

        # Save model to a temp file and offer download
        tmpdir = tempfile.gettempdir()
        model_path = os.path.join(tmpdir, download_model_name)
        model.save(model_path)
        st.success(f"Model saved to temporary path: {model_path}")

        # Provide a download button
        with open(model_path, "rb") as f:
            st.download_button("Download model (.keras)", data=f, file_name=download_model_name)

        # Show basic loss curve if available
        if 'loss' in history.history:
            fig3, ax3 = plt.subplots(figsize=(6, 3))
            ax3.plot(history.history['loss'], label='loss')
            ax3.set_title("Training Loss")
            ax3.set_xlabel("Epoch")
            ax3.set_ylabel("Loss")
            ax3.legend()
            st.pyplot(fig3)

# ---------------------------
# Quick predict using last trained model on full data (optional)
# ---------------------------
st.markdown("---")
st.write("### Quick predict (use existing saved model file)")

uploaded_model = st.file_uploader("Upload `.keras` model file (optional)", type=["keras"])
predict_button = st.button("Run quick prediction (last lookback -> next)")

loaded_model = None
if uploaded_model is not None:
    tmp_mod = os.path.join(tempfile.gettempdir(), uploaded_model.name)
    with open(tmp_mod, "wb") as f:
        f.write(uploaded_model.getbuffer())
    try:
        if TF_AVAILABLE:
            loaded_model = load_model(tmp_mod)
            st.success("Model loaded from upload.")
        else:
            st.error("Uploaded model cannot be loaded because TensorFlow is not available in this environment.")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        loaded_model = None
else:
    local_default = os.path.join(os.getcwd(), download_model_name)
    if os.path.exists(local_default) and TF_AVAILABLE:
        try:
            loaded_model = load_model(local_default)
            st.info(f"Loaded model from {local_default}")
        except Exception as e:
            st.warning(f"Could not load local model: {e}")
            loaded_model = None

if predict_button:
    if loaded_model is None:
        st.error("No model available. Train a model or upload a .keras file first.")
    else:
        st.info("Preparing latest data and predicting next day(s)...")
        prices_full = df['Close'].values
        scaler_tmp = MinMaxScaler(feature_range=(0, 1))
        scaler_tmp.fit(prices_full.reshape(-1, 1))
        last_window = prices_full[-lookback:].reshape(-1, 1)
        last_scaled = scaler_tmp.transform(last_window)
        X_pred = last_scaled.reshape((1, last_scaled.shape[0], 1))
        pred_scaled = loaded_model.predict(X_pred)
        pred = inverse_scale(pred_scaled.flatten(), scaler_tmp)[0]
        st.success(f"Next predicted price (one-step): {pred:.2f}")
        st.write("Note: This quick prediction uses a full-data-fitted scaler for simplicity. For production, save and reuse the training scaler.")

st.markdown("---")
st.caption("Built from a notebook LSTM pipeline. This demo is for learning / prototyping. Do not use for trading decisions.")

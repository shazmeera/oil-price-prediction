import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import pickle

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense

# =========================
# LOAD DATA
# =========================
df = pd.read_excel("dataset.xlsx")
df.columns = df.columns.str.strip()

st.title("📊 AI Oil Price Prediction System (PRO LEVEL)")

# =========================
# FEATURES
# =========================
features = df[[
    "CPI",
    "ExchangeRate",
    "IndustrialIndex",
    "oil Imports (million USD)",
    "OilPrice"
]]

# =========================
# SCALING
# =========================
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(features)

# =========================
# CREATE SEQUENCES (LSTM)
# =========================
X, y = [], []

for i in range(10, len(scaled_data)):
    X.append(scaled_data[i-10:i])
    y.append(scaled_data[i, 4])  # OilPrice column

X, y = np.array(X), np.array(y)

# =========================
# LOAD OR TRAIN MODEL
# =========================
try:
    model = load_model("oil_lstm.h5")
    scaler = pickle.load(open("scaler.pkl", "rb"))
    st.success("Model Loaded Successfully")
except:
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(100))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, batch_size=16, verbose=0)

    model.save("oil_lstm.h5")
    pickle.dump(scaler, open("scaler.pkl", "wb"))

    st.success("Model Trained & Saved")

# =========================
# STEP 9: CHARTS
# =========================

st.subheader("📈 Oil Price Trend")

fig = px.line(df, x=df.index, y="OilPrice", title="Oil Price Trend")
st.plotly_chart(fig)

# Moving Average
df["MA"] = df["OilPrice"].rolling(5).mean()

fig2 = px.line(df, x=df.index, y=["OilPrice", "MA"],
               title="Moving Average Trend")
st.plotly_chart(fig2)

# =========================
# STEP 10: LIVE DATA
# =========================
st.subheader("🌍 Live Data")

try:
    usd = requests.get("https://api.exchangerate-api.com/v4/latest/USD").json()
    st.metric("USD to PKR", usd["rates"]["PKR"])
except:
    st.warning("Live data not available")

# =========================
# STEP 11: NEXT 30 DAYS FORECAST
# =========================

st.subheader("🔮 Next 30 Days Forecast")

future_predictions = []

last_10 = scaled_data[-10:]

for i in range(30):  # 30 days forecast
    input_seq = last_10.reshape(1, 10, X.shape[2])

    pred = model.predict(input_seq, verbose=0)[0][0]

    future_predictions.append(pred)

    new_row = last_10[-1].copy()
    new_row[4] = pred  # replace oil price
    last_10 = np.append(last_10[1:], [new_row], axis=0)

# Convert back to real values
future_prices = np.array(future_predictions).reshape(-1, 1)

dummy = np.zeros((future_prices.shape[0], features.shape[1]))
dummy[:, 4] = future_prices[:, 0]

real_prices = scaler.inverse_transform(dummy)[:, 4]

# Create forecast dataframe
forecast_df = pd.DataFrame({
    "Day": np.arange(1, 31),
    "Predicted Oil Price": real_prices
})

st.line_chart(forecast_df.set_index("Day"))

st.dataframe(forecast_df)

# =========================
# USER INPUT PREDICTION
# =========================
st.subheader("📌 Manual Prediction")

cpi = st.number_input("CPI", value=280.0)
exchange = st.number_input("Exchange Rate", value=300.0)
industrial = st.number_input("Industrial Index", value=120.0)
imports = st.number_input("Oil Imports", value=400.0)

if st.button("Predict Now"):
    input_data = np.array([[cpi, exchange, industrial, imports, 100]])

    input_scaled = scaler.transform(input_data.reshape(1, -1))
    input_scaled = input_scaled.reshape(1, 1, input_scaled.shape[1])

    prediction = model.predict(input_scaled, verbose=0)

    st.success(f"Predicted Oil Price: {prediction[0][0]:.2f}")

# =========================
# MODEL INFO
# =========================
st.subheader("📊 Model Summary")
st.write("LSTM Deep Learning Model with Multi-Feature Time Series")
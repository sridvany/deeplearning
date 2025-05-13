import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

st.title("📈 LSTM ile Hisse Fiyat Tahmini (Streamlit Uygulaması)")

symbol = st.selectbox("Hisse Sembolü Seçin:", ['AKBNK.IS', 'GARAN.IS'])

start_date = st.date_input("Başlangıç Tarihi", pd.to_datetime("2005-01-01"))
end_date = st.date_input("Bitiş Tarihi", pd.to_datetime("2025-01-01"))

if st.button("Modeli Eğit ve Tahmin Et"):
    data = yf.download(symbol, start=start_date, end=end_date)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].astype('float64')
    st.write("Veri Önizleme:", data.tail())

    features = ['Open', 'High', 'Low', 'Volume']
    X = data[features].values
    y = data['Close'].values.reshape(-1, 1)

    train_end = int(len(data) * 0.7)
    val_end = train_end + int(len(data) * 0.15)

    X_train_raw, y_train_raw = X[:train_end], y[:train_end]
    X_val_raw, y_val_raw = X[train_end:val_end], y[train_end:val_end]
    X_test_raw, y_test_raw = X[val_end:], y[val_end:]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train = scaler_X.fit_transform(X_train_raw)
    y_train = scaler_y.fit_transform(y_train_raw)
    X_val = scaler_X.transform(X_val_raw)
    y_val = scaler_y.transform(y_val_raw)
    X_test = scaler_X.transform(X_test_raw)
    y_test = scaler_y.transform(y_test_raw)

    def create_sequences(X, y, window_size):
        Xs, ys = [], []
        for i in range(window_size, len(X)):
            Xs.append(X[i - window_size:i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    def create_lstm_model(num_layers, input_shape):
        model = Sequential()
        for i in range(num_layers):
            units = 64 // (2 ** i)
            return_sequences = True if i < num_layers - 1 else False
            if i == 0:
                model.add(LSTM(units, return_sequences=return_sequences, input_shape=input_shape))
            else:
                model.add(LSTM(units, return_sequences=return_sequences))
            model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model

    window_size = 30
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, window_size)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, window_size)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, window_size)

    model = create_lstm_model(2, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]))
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(X_train_seq, y_train_seq,
                        epochs=20, batch_size=32,
                        validation_data=(X_val_seq, y_val_seq),
                        callbacks=[early_stop], verbose=0)

    y_pred = model.predict(X_test_seq)
    y_pred_rescaled = scaler_y.inverse_transform(y_pred)
    y_test_rescaled = scaler_y.inverse_transform(y_test_seq)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test_rescaled, label='Gerçek')
    ax.plot(y_pred_rescaled, label='Tahmin')
    ax.set_title(f"{symbol} Tahmin Sonuçları")
    ax.set_xlabel("Zaman")
    ax.set_ylabel("Fiyat")
    ax.legend()
    st.pyplot(fig)

    rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    r2 = r2_score(y_test_rescaled, y_pred_rescaled)

    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**R2:** {r2:.2f}")

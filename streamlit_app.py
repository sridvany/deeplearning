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

st.set_page_config(layout="wide")
st.title("📊 LSTM ile Hisse Senedi Fiyat Tahmini")

# Kullanıcıdan sembol seçimi
symbols_default = ['AKBNK.IS', 'GARAN.IS']
symbols = st.multiselect("Tahmin yapılacak hisseleri seçin:", options=symbols_default, default=symbols_default)

# Hiperparametre ayarları
with st.expander("⚙️ Hiperparametreleri Ayarla"):
    units1 = st.slider("Katman 1 Nöron Sayısı", 32, 512, 256, 32)
    units2 = st.slider("Katman 2 Nöron Sayısı", 16, 256, 128, 16)
    units3 = st.slider("Katman 3 Nöron Sayısı", 8, 128, 64, 8)
    units4 = st.slider("Katman 4 Nöron Sayısı", 4, 64, 32, 4)
    dropout = st.slider("Dropout Oranı", 0.0, 0.5, 0.2, 0.05)
    window_size = st.slider("Geçmiş Gün Sayısı (Sliding Window)", 10, 100, 30)
    batch_size = st.selectbox("Batch Size", [32, 64, 128])
    epochs = st.slider("Epoch Sayısı", 10, 300, 100, 10)
    patience = st.slider("Erken Durdurma (patience)", 5, 20, 10)

# LSTM modeli oluşturucu
def create_lstm_model(num_layers, input_shape, hyperparams):
    model = Sequential()
    for i in range(num_layers):
        units = hyperparams[f'units{i + 1}']
        return_sequences = i < num_layers - 1
        if i == 0:
            model.add(LSTM(units, return_sequences=return_sequences, input_shape=input_shape))
        else:
            model.add(LSTM(units, return_sequences=return_sequences))
        model.add(Dropout(hyperparams['dropout']))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Sekans oluşturucu
def create_sequences(X, y, window_size):
    Xs, ys = [], []
    for i in range(window_size, len(X) - 1):
        Xs.append(X[i - window_size:i])
        ys.append(y[i + 1])
    return np.array(Xs), np.array(ys)

if st.button("📈 Modeli Eğit ve Tahmin Yap"):
    results_all_symbols = {}

    for symbol in symbols:
        st.subheader(f"📌 {symbol} Analizi")
        data = yf.download(symbol, start='2005-01-01', end='2025-01-01')
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].astype('float64')

        features = ['Open', 'High', 'Low', 'Volume']
        X = data[features].values
        y = data['Close'].values.reshape(-1, 1)

        # Eğitim/Doğrulama/Test bölme
        total_samples = len(data)
        train_end = int(total_samples * 0.7)
        val_end = train_end + int(total_samples * 0.15)
        X_train_raw, X_val_raw, X_test_raw = X[:train_end], X[train_end:val_end], X[val_end:]
        y_train_raw, y_val_raw, y_test_raw = y[:train_end], y[train_end:val_end], y[val_end:]

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_train = scaler_X.fit_transform(X_train_raw)
        y_train = scaler_y.fit_transform(y_train_raw)
        X_val = scaler_X.transform(X_val_raw)
        y_val = scaler_y.transform(y_val_raw)
        X_test = scaler_X.transform(X_test_raw)
        y_test = scaler_y.transform(y_test_raw)

        X_train_seq, y_train_seq = create_sequences(X_train, y_train, window_size)
        X_val_seq, y_val_seq = create_sequences(X_val, y_val, window_size)
        X_test_seq, y_test_seq = create_sequences(X_test, y_test, window_size)

        hyperparams_dict = {
            'units1': units1, 'units2': units2, 'units3': units3, 'units4': units4,
            'dropout': dropout, 'batch_size': batch_size, 'epochs': epochs,
            'patience': patience, 'window_size': window_size
        }

        symbol_results = {}
        for num_layers in [1, 2, 3, 4]:
            model = create_lstm_model(num_layers, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]), hyperparams=hyperparams_dict)
            early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
            model.fit(X_train_seq, y_train_seq, epochs=epochs, batch_size=batch_size,
                      validation_data=(X_val_seq, y_val_seq), callbacks=[early_stop], verbose=0)

            y_pred = model.predict(X_test_seq)
            rmse = np.sqrt(mean_squared_error(y_test_seq, y_pred))
            mae = mean_absolute_error(y_test_seq, y_pred)
            mape = mean_absolute_percentage_error(y_test_seq, y_pred)
            r2 = r2_score(y_test_seq, y_pred)
            symbol_results[num_layers] = {
                'rmse': rmse, 'mae': mae, 'mape': mape, 'r2': r2,
                'y_pred': y_pred, 'y_true': y_test_seq
            }

        # Tablo ve grafik gösterimi
        results_df = pd.DataFrame([
            {'Katman': k, 'RMSE': v['rmse'], 'MAE': v['mae'], 'MAPE': v['mape'], 'R²': v['r2']}
            for k, v in symbol_results.items()
        ])
        st.dataframe(results_df.style.format(precision=4))

        for i, num_layers in enumerate([1, 2, 3, 4]):
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(symbol_results[num_layers]['y_true'], label='Gerçek')
            ax.plot(symbol_results[num_layers]['y_pred'], label='Tahmin')
            ax.set_title(f"{symbol} - {num_layers} Katmanlı LSTM")
            ax.set_xlabel("Zaman")
            ax.set_ylabel("Scaled Close")
            ax.legend()
            st.pyplot(fig)

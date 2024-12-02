import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


# Veri setini yükleme
data1 = pd.read_csv('C:\\Users\\LENOVO\\Desktop\\ltsm\\LSTM-Multivariate_pollution.csv')
print(data1.columns)

# İlk birkaç satırı kontrol et
print(data1.head())

# 1. Veri Hazırlığı
features = ['pollution', 'dew', 'temp', 'press', 'wnd_spd', 'snow', 'rain']
data_lstm = data1[features].copy()

# Verileri normalize et
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_lstm)

# Lag özellikleri oluştur
def create_sequences(data, sequence_length=24):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length, :])
        y.append(data[i + sequence_length, 0])  # Hedef: pollution
    return np.array(X), np.array(y)

sequence_length = 24
X, y = create_sequences(data_scaled, sequence_length)

# Eğitim ve test verilerini ayır
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# 2. Model Tasarımı
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
    Dense(32, activation='relu'),
    Dense(1)  # Çıktı katmanı
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# 3. Model Eğitimi
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32,
    verbose=1
)

# 4. Değerlendirme ve Görselleştirme
# Tahminler
y_pred = model.predict(X_test)

# Normalize edilmiş verileri geri çevir
y_test_rescaled = scaler.inverse_transform(
    np.concatenate([y_test.reshape(-1, 1), np.zeros((len(y_test), X_test.shape[2]-1))], axis=1)
)[:, 0]

y_pred_rescaled = scaler.inverse_transform(
    np.concatenate([y_pred.reshape(-1, 1), np.zeros((len(y_pred), X_test.shape[2]-1))], axis=1)
)[:, 0]

# Sonuçları görselleştir
plt.figure(figsize=(10, 6))
plt.plot(y_test_rescaled[:100], label='Gerçek PM2.5', color='blue')
plt.plot(y_pred_rescaled[:100], label='Tahmin PM2.5', color='red')
plt.title('PM2.5 Tahmini (LSTM)')
plt.xlabel('Zaman')
plt.ylabel('PM2.5')
plt.legend()
plt.show()

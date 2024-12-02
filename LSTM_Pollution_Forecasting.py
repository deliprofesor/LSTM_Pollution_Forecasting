#Kitiphanler yüklendi
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Veri setini yükleme
data1 = pd.read_csv('LSTM-Multivariate_pollution.csv')
print(data1.columns)

# İlk birkaç satırı kontrol et
print(data1.head())  # Verinin ilk 5 satırını yazdır

# 1. Veri Hazırlığı
features = ['pollution', 'dew', 'temp', 'press', 'wnd_spd', 'snow', 'rain']  # Kullanılacak özellikler
data_lstm = data1[features].copy()  # Seçilen özelliklerle veri oluştur

# Verileri normalize et (0-1 arasında ölçekleme)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_lstm)

# Lag özellikleri oluştur (zaman serisi verisi için ardışık gözlemler)
def create_sequences(data, sequence_length=24):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length, :])  # Girdi verileri
        y.append(data[i + sequence_length, 0])  # Hedef: pollution (ilk sütun)
    return np.array(X), np.array(y)

sequence_length = 24  # Girdi dizisinin uzunluğu
X, y = create_sequences(data_scaled, sequence_length)  # Veri setini ardışık dizilere dönüştür

# Eğitim ve test verilerini ayır (80% eğitim, 20% test)
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]  # Eğitim ve test verileri
y_train, y_test = y[:split_index], y[split_index:]  # Eğitim ve test etiketleri

# 2. Model Tasarımı
model = Sequential([  # Modelin oluşturulması
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),  # LSTM katmanı
    Dense(32, activation='relu'),  # Tam bağlı katman
    Dense(1)  # Çıktı katmanı (sadece bir değer: pollution tahmini)
])

# Modeli derleme (optimizer: Adam, loss: mean squared error)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# 3. Model Eğitimi
history = model.fit(
    X_train, y_train,  # Eğitim verileri
    validation_data=(X_test, y_test),  # Test verileri ile doğrulama
    epochs=20,  # Eğitim dönemi sayısı
    batch_size=32,  # Her bir eğitim adımındaki örnek sayısı
    verbose=1  # Eğitim sürecinin çıktı olarak gösterilmesi
)

# 4. Değerlendirme ve Görselleştirme
# Test verisi üzerinde tahminler
y_pred = model.predict(X_test)

# Normalize edilmiş verileri geri çevir (gerçek ve tahmin edilen sonuçlar)
y_test_rescaled = scaler.inverse_transform(
    np.concatenate([y_test.reshape(-1, 1), np.zeros((len(y_test), X_test.shape[2]-1))], axis=1)
)[:, 0]

y_pred_rescaled = scaler.inverse_transform(
    np.concatenate([y_pred.reshape(-1, 1), np.zeros((len(y_pred), X_test.shape[2]-1))], axis=1)
)[:, 0]

# Sonuçları görselleştir (ilk 100 tahmini ve gerçek veriyi karşılaştır)
plt.figure(figsize=(10, 6))
plt.plot(y_test_rescaled[:100], label='Gerçek PM2.5', color='blue')  # Gerçek veriler
plt.plot(y_pred_rescaled[:100], label='Tahmin PM2.5', color='red')  # Tahmin edilen veriler
plt.title('PM2.5 Tahmini (LSTM)')  # Grafik başlığı
plt.xlabel('Zaman')  # X ekseni etiketi
plt.ylabel('PM2.5')  # Y ekseni etiketi
plt.legend()  # Grafik üzerinde etiketler
plt.show()  # Grafiği göster

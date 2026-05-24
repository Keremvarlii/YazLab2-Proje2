import pandas as pd
import os
import glob

def skab_verisini_hazirla():
    tum_veriler = []
    
    # valve1 ve valve2 klasörlerini sırayla geziyoruz
    klasorler = ["valve1", "valve2"]
    
    for klasor in klasorler:
        # Klasörün içindeki tüm .csv uzantılı dosyaları buluyoruz
        dosya_yollari = glob.glob(os.path.join(klasor, "*.csv"))
        
        for dosya in dosya_yollari:
            # CSV dosyasını pandas ile okuyoruz (SKAB genelde noktalı virgül ile ayrılır)
            df = pd.read_csv(dosya, sep=";") 
            
            # Eğer sep=";" hata verirse, sadece pd.read_csv(dosya) olarak değiştirebiliriz.
            # Bazen SKAB verileri indekssiz gelir, ilk sütun datetime'dır.
            
            # Hocanın istediği yeni sütunları ekliyoruz
            df["source_group"] = klasor
            df["source_file"] = os.path.basename(dosya)
            
            # Okuduğumuz ve sütun eklediğimiz bu tabloyu listeye atıyoruz
            tum_veriler.append(df)
            
    # Listedeki tüm tabloları alt alta birleştirip tek bir DataFrame yapıyoruz
    birlestirilmis_skab = pd.concat(tum_veriler, ignore_index=True)
    return birlestirilmis_skab

# Fonksiyonu çalıştırıp test edelim
skab_veri = skab_verisini_hazirla()
print("SKAB verisi başarıyla birleştirildi!")
print("Toplam Satır ve Sütun Sayısı:", skab_veri.shape)
print("\nİlk 5 Satır:")
print(skab_veri.head())

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# SKAB verisinden modele girmeyecek sütunları çıkarıyoruz 
dusulecek_sutunlar = ['datetime', 'changepoint', 'source_group', 'source_file', 'anomaly']

# X (Girdiler - Sensörler) ve y (Hedef - Anomali)
X = skab_veri.drop(columns=dusulecek_sutunlar)
y = skab_veri['anomaly'] 
groups = skab_veri['source_file'] # Dosya bazlı bölme için gruplarımız

# Pipeline: Önce veriyi ölçeklendir (Normalize), sonra PCA ile 1 boyuta indir 
on_isleme_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=1))
])

# KFold ile veriyi 5 parçaya bölüyoruz 
gkf = GroupKFold(n_splits=5)

# İlk parçalanmayı test edelim
for train_idx, test_idx in gkf.split(X, y, groups=groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Pipeline SADECE train verisiyle eğitilir (Fit), sonra test verisine dönüştürülür (Transform) 
    X_train_islenmis = on_isleme_pipeline.fit_transform(X_train)
    X_test_islenmis = on_isleme_pipeline.transform(X_test)
    
    print("\nVeri başarıyla bölündü ve işlendi!")
    print(f"Eğitim verisi boyutu (PCA sonrası): {X_train_islenmis.shape}")
    print(f"Test verisi boyutu (PCA sonrası): {X_test_islenmis.shape}")
    break # Sadece ilk bölmeyi test etmek için döngüyü şimdilik kırıyoruz

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Kodu her çalıştırdığımızda aynı sonucu almak için hocanın istediği seed değerlerinden birini ayarlayalım [cite: 103]
tf.random.set_seed(42)
np.random.seed(42)

# Zaman serisi verisini 4'lük pencerelere (sekanslara) bölme işlemi
def sekans_olustur(X, y, pencere_boyutu=4):
    X_seq, y_seq = [], []
    for i in range(len(X) - pencere_boyutu):
        X_seq.append(X[i:(i + pencere_boyutu)])
        y_seq.append(y.iloc[i + pencere_boyutu]) 
    return np.array(X_seq), np.array(y_seq)

# Verimizi pencerelere bölüyoruz
X_train_seq, y_train_seq = sekans_olustur(X_train_islenmis, y_train, pencere_boyutu=4)
X_test_seq, y_test_seq = sekans_olustur(X_test_islenmis, y_test, pencere_boyutu=4)

# 1. MODEL: LSTM
lstm_model = Sequential([
    LSTM(32, input_shape=(4, 1)), # 4 zaman adımı, 1 özellik (PCA'dan gelen)
    Dense(1, activation='sigmoid') # 0 (Normal) veya 1 (Anomali) çıktısı
])

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Hocanın zorunlu kıldığı "Erken Durdurma" (Early Stopping) ayarı 
erken_durma = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("LSTM Modeli eğitime başlıyor... (Ekranda yazılar akacak)")

# Modeli Eğitiyoruz (Hocanın kuralı: max epoch 50, batch_size 32 [cite: 100, 101])
history_lstm = lstm_model.fit(
    X_train_seq, y_train_seq,
    validation_split=0.2, # Doğrulama için verinin %20'si
    epochs=50,
    batch_size=32,
    callbacks=[erken_durma],
    verbose=1
)

print("\nLSTM Eğitimi Başarıyla Tamamlandı!")

from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten

# 2. MODEL: 1D-CNN
cnn_model = Sequential([
    Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(4, 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(1, activation='sigmoid') # Yine 0 (Normal) veya 1 (Anomali) çıktısı
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\n1D-CNN Modeli eğitime başlıyor...")

# Aynı erken durma (erken_durma) ayarını burada da kullanıyoruz
history_cnn = cnn_model.fit(
    X_train_seq, y_train_seq,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[erken_durma], 
    verbose=1
)

print("\n1D-CNN Eğitimi Başarıyla Tamamlandı!")
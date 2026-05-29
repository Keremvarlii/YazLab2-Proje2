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

from pyts.approximation import SymbolicAggregateApproximation
import warnings
warnings.filterwarnings('ignore') # Terminali kalabalıklaştırmaması için gereksiz uyarıları gizliyoruz

# 3. MODEL: OTOMATA (AÇIKLANABİLİR MODEL)
print("\nOtomata Modeli için veriler sembollere (SAX) çevriliyor...")

# Hocanın zorunlu tuttuğu sabit parametreler
pencere_boyutu = 4
alfabe_boyutu = 3

# Modeller için ayırdığımız X_train_seq boyutumuz (N, 4, 1) şeklindeydi. 
# SAX dönüşümü için bunu (N, 4) formuna getiriyoruz.
X_train_2d = X_train_seq.reshape(-1, pencere_boyutu)
X_test_2d = X_test_seq.reshape(-1, pencere_boyutu)

# SAX dönüştürücüsünü tanımlıyoruz (PAA işlemini kendi içinde otomatik yapar)
sax = SymbolicAggregateApproximation(n_bins=alfabe_boyutu, strategy='normal')

# SADECE train verisi ile fit ediyoruz (Veri sızıntısını önlemek için hocanın kuralı)
X_train_sax_dizileri = sax.fit_transform(X_train_2d)
X_test_sax_dizileri = sax.transform(X_test_2d)

# Çıkan harfleri (örneğin ['a', 'a', 'b', 'c']) tek bir string kelimeye ('aabc') çeviren yardımcı fonksiyon
def kelimeye_cevir(sax_matrisi):
    kelimeler = []
    for satir in sax_matrisi:
        # pyts 'a', 'b', 'c' gibi harfleri üretir, bunları birleştirip pattern yapıyoruz
        kelime = "".join(satir)
        kelimeler.append(kelime)
    return kelimeler

train_durumlar = kelimeye_cevir(X_train_sax_dizileri)
test_durumlar = kelimeye_cevir(X_test_sax_dizileri)

print(f"Eğitim verisinden elde edilen ilk 5 durum (Pattern): {train_durumlar[:5]}")

from collections import defaultdict

print("\nDurumlar arası geçiş olasılıkları hesaplanıyor...")

# Geçişleri saymak için sözlük (dictionary) yapıları oluşturuyoruz
gecis_sayilari = defaultdict(lambda: defaultdict(int))
cikis_sayilari = defaultdict(int)

# Eğitim verisindeki (train_durumlar) örüntüleri sırayla gezip kimden kime gidilmiş sayıyoruz
for i in range(len(train_durumlar) - 1):
    mevcut_durum = train_durumlar[i]
    sonraki_durum = train_durumlar[i+1]
    
    gecis_sayilari[mevcut_durum][sonraki_durum] += 1
    cikis_sayilari[mevcut_durum] += 1

# Sayıları olasılıklara (0 ile 1 arasında değerlere) çeviriyoruz
gecis_olasiliklari = defaultdict(dict)

for durum, gecisler in gecis_sayilari.items():
    for sonraki, sayi in gecisler.items():
        # Formül: (A'dan B'ye geçiş sayısı) / (A'dan yapılan toplam çıkış sayısı)
        olasilik = sayi / cikis_sayilari[durum]
        gecis_olasiliklari[durum][sonraki] = round(olasilik, 4)

print(f"Olasılıklar Hesaplandı! Sistemde toplam {len(gecis_olasiliklari)} farklı benzersiz durum (state) öğrenildi.")

# Az önce ekranda gördüğümüz 'cccc' durumunun nereye gittiğine (olasılıklarına) bakalım:
if 'cccc' in gecis_olasiliklari:
    print(f"\nÖrnek İnceleme -> 'cccc' durumundan sonraki hedefler ve olasılıkları:")
    print(gecis_olasiliklari['cccc'])

    import Levenshtein
import json

print("\nAçıklanabilirlik Modülü Çalıştırılıyor...")

# Unseen pattern'lar için en yakın durumu bulan fonksiyon
def en_yakin_durumu_bul(unseen_pattern, bilinen_durumlar):
    min_mesafe = float('inf')
    en_yakin = None
    for durum in bilinen_durumlar:
        mesafe = Levenshtein.distance(unseen_pattern, durum)
        if mesafe < min_mesafe:
            min_mesafe = mesafe
            en_yakin = durum
    return en_yakin, min_mesafe

# Test verisinden hocanın istediği formata uygun bir anı (örneğin 5. zaman adımı) çekelim
zaman_adimi = 5
onceki_durum = test_durumlar[zaman_adimi - 1]
mevcut_pattern = test_durumlar[zaman_adimi]

durum_statu = "seen"
eslesen_pattern = mevcut_pattern

# Sistemdeki bilinen durumları listeliyoruz
bilinen_durumlar = list(gecis_olasiliklari.keys())

# Eğer test verisindeki pattern eğitimde yoksa (Unseen ise)
if mevcut_pattern not in bilinen_durumlar:
    durum_statu = "unseen"
    eslesen_pattern, mesafe = en_yakin_durumu_bul(mevcut_pattern, bilinen_durumlar)

# Olasılık Hesabı (Önceki durumdan, eşleşen duruma geçiş ihtimali)
# Eğer böyle bir geçiş hiç yaşanmamışsa, sistem sıfıra bölme hatası vermesin diye çok küçük bir ihtimal (0.0001) atıyoruz
gecis_olasiligi = gecis_olasiliklari.get(onceki_durum, {}).get(eslesen_pattern, 0.0001)

# Karar Mekanizması: İhtimal %5'ten (0.05) düşükse sistem bunu beklemiyordur, yani bu bir ANOMALİDİR.
karar = "anomaly" if gecis_olasiligi < 0.05 else "normal"

# Hocanın raporda zorunlu tuttuğu JSON Çıktısı Formatı
aciklama_ciktisi = {
    "time_step": zaman_adimi,
    "state": onceki_durum,
    "pattern": mevcut_pattern,
    "status": durum_statu,
    "mapped_to": eslesen_pattern,
    "probability": gecis_olasiligi,
    "decision": karar
}

print("\n--- [SYSTEM DECISION] ---")
print(json.dumps(aciklama_ciktisi, indent=4))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("\n--- 4. GÜRÜLTÜ (NOISE) VE DEĞERLENDİRME AŞAMASI ---")

# Gürültü ekleme (Gaussian Noise)
# Test verimize biraz 'parazit' katıyoruz ki modellerin ne kadar sağlam durduğunu görelim
gurultu_orani = 0.5 
X_test_gurultulu = X_test_islenmis + np.random.normal(0, gurultu_orani, X_test_islenmis.shape)

# Gürültülü veriyi pencerelere (sekanslara) bölüyoruz (LSTM için)
X_test_gurultulu_seq, y_test_gurultulu_seq = sekans_olustur(X_test_gurultulu, y_test, pencere_boyutu=4)

# Değerlendirme fonksiyonu (Metrikleri hesaplamak için)
def metrikleri_hesapla(y_gercek, y_tahmin_olasilik, model_adi, senaryo="Orijinal"):
    # Olasılıkları 0 veya 1'e yuvarlıyoruz (0.5 üstü anomali kabul edilir)
    y_tahmin = (y_tahmin_olasilik > 0.5).astype(int)
    
    acc = accuracy_score(y_gercek, y_tahmin)
    prec = precision_score(y_gercek, y_tahmin, zero_division=0)
    rec = recall_score(y_gercek, y_tahmin, zero_division=0)
    f1 = f1_score(y_gercek, y_tahmin, zero_division=0)
    
    print(f"\n[{model_adi}] - Senaryo: {senaryo}")
    print(f"Accuracy : %{acc*100:.2f}")
    print(f"Precision: %{prec*100:.2f}")
    print(f"Recall   : %{rec*100:.2f}")
    print(f"F1-Score : %{f1*100:.2f}")

# LSTM Modeli için tahminler (Orijinal test verisi)
lstm_tahminler_orijinal = lstm_model.predict(X_test_seq, verbose=0)
metrikleri_hesapla(y_test_seq, lstm_tahminler_orijinal, "LSTM", "Orijinal Veri")

# LSTM Modeli için tahminler (Gürültülü test verisi)
lstm_tahminler_gurultulu = lstm_model.predict(X_test_gurultulu_seq, verbose=0)
metrikleri_hesapla(y_test_gurultulu_seq, lstm_tahminler_gurultulu, "LSTM", "Gürültülü (Noise) Veri")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

print("\n--- 5. GÖRSELLEŞTİRME VE GRAFİKLER ---")

# Grafiklerin düzgün görünmesi için genel bir boyut ayarlayalım
plt.figure(figsize=(18, 5))

# 1. Confusion Matrix (Karmaşıklık Matrisi)
plt.subplot(1, 3, 1)
y_tahmin_orijinal = (lstm_tahminler_orijinal > 0.5).astype(int)
cm = confusion_matrix(y_test_seq, y_tahmin_orijinal)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("LSTM - Confusion Matrix (Orijinal)")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek Durum")

# 2. ROC Eğrisi
plt.subplot(1, 3, 2)
fpr, tpr, thresholds = roc_curve(y_test_seq, lstm_tahminler_orijinal)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title("LSTM - ROC Eğrisi")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")

# 3. Transition Probability Heatmap (İlk 10 Durum İçin)
# Çok fazla durum (81) olduğu için grafiğe sığsın diye en çok tekrar eden 10 tanesini alıyoruz
plt.subplot(1, 3, 3)
en_cok_cikanlar = sorted(cikis_sayilari.keys(), key=lambda x: cikis_sayilari[x], reverse=True)[:10]
isi_haritasi_verisi = np.zeros((10, 10))

for i, d1 in enumerate(en_cok_cikanlar):
    for j, d2 in enumerate(en_cok_cikanlar):
        isi_haritasi_verisi[i, j] = gecis_olasiliklari.get(d1, {}).get(d2, 0.0)

sns.heatmap(isi_haritasi_verisi, annot=True, fmt=".2f", cmap="YlGnBu", 
            xticklabels=en_cok_cikanlar, yticklabels=en_cok_cikanlar)
plt.title("Otomata Geçiş Olasılıkları (İlk 10 Durum)")
plt.xlabel("Sonraki Durum")
plt.ylabel("Mevcut Durum")

plt.tight_layout()
plt.show() # Grafikleri ekranda göster
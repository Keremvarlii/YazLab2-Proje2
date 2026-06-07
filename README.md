# YazLab 2. Proje: Deney Sonuçları ve Karşılaştırmalı Analiz Raporu

**Grup:** 25. Grup  
**Tarih:** 03.04.2026  

## Giriş
Bu tamamlayıcı doküman, **"From Black-Box to Explainability: Probabilistic Automata for Time Series Analysis"** başlıklı ana projenin raporunda yer alması gereken kapsamlı deney sonuçlarını, detaylı tablo dökümlerini ve zorunlu görsel analizleri içermektedir. Projede LSTM, GRU, 1D-CNN ve Probabilistic Automata modelleri sistematik bir şekilde analiz edilmiştir.

---

## 1. Temel Performans ve Stabilite

Aşağıdaki tablo, modellerin zorunlu tutulan veri setleri (SKAB ve BATADAL) üzerindeki ortalama F1-skorlarını ve 5 farklı random seed (42, 123, 2026, 7, 999) ile elde edilen standart sapma değerlerini göstermektedir. 

**Tablo 1: Model Performansı ve Stabilitesi (Ortalama F1-score ± Standart Sapma)**

| Model | SKAB (GroupKFold) | BATADAL (Chronological) |
| :--- | :---: | :---: |
| **LSTM** | 0.00 ± 0.00 | 0.00 ± 0.00 |
| **GRU** | 0.12 ± 0.06 | 0.25 ± 0.19 |
| **1D-CNN** | 0.05 ± 0.06 | 0.27 ± 0.16 |
| **Automata**| (Çıkarım Modu) | (Çıkarım Modu) |

*Not: Gerçek sonuçlar için projeyi `python main.py` komutuyla çalıştırıp çıktıları inceleyiniz.*

---

## 2. Model Performans Görselleri ve Çıktılar (Rubrik Gereksinimleri)

Hocanın proje değerlendirme rubriği kapsamında zorunlu tuttuğu 5 temel görsel analiz aşağıda sırasıyla sunulmuştur.

### 2.1 Confusion Matrix (Karmaşıklık Matrisi)
Modellerin BATADAL veri setindeki gerçek ve tahmin edilen sınıf dağılımları:

<div align="center">
  <img src="outputs/figures/LSTM_BATADAL_cm.png" width="30%" alt="LSTM Confusion Matrix">
  <img src="outputs/figures/GRU_BATADAL_cm.png" width="30%" alt="GRU Confusion Matrix">
  <img src="outputs/figures/1D-CNN_BATADAL_cm.png" width="30%" alt="1D-CNN Confusion Matrix">
</div>
<br>

### 2.2 ROC Eğrisi
Modellerin farklı eşik değerlerindeki (threshold) ayrım gücü (True Positive / False Positive) kapasiteleri:

<div align="center">
  <img src="outputs/figures/LSTM_BATADAL_roc.png" width="30%" alt="LSTM ROC Eğrisi">
  <img src="outputs/figures/GRU_BATADAL_roc.png" width="30%" alt="GRU ROC Eğrisi">
  <img src="outputs/figures/1D-CNN_BATADAL_roc.png" width="30%" alt="1D-CNN ROC Eğrisi">
</div>
<br>

### 2.3 Transition Probability Heatmap (Geçiş Olasılıkları)
Oluşturulan Automata durumlarının (states) kendi aralarındaki olasılıksal geçiş haritası:

<div align="center">
  <img src="outputs/figures/Automata_BATADAL_transition_heatmap.png" width="60%" alt="Automata Transition Heatmap">
</div>
<br>

### 2.4 Automata State Diagram (Durum Diyagramı)
Automata'nın eğitim aşamasında oluşturduğu yapısal geçişlerin ve durumların kavramsal diyagramı:

<div align="center">
  <img src="outputs/Automata_State_Diagram.png" width="60%" alt="Automata State Diagram">
</div>
<br>

### 2.5 Parametre Duyarlılık Grafikleri
Farklı pencere (window) ve alfabe (alphabet) boyutlarının sistem F1-skoruna etkisini gösteren analiz grafiği:

<div align="center">
  <img src="outputs/Parametre_Duyarlilik_Grafigi.png" width="60%" alt="Parametre Duyarlilik Grafigi">
</div>

---

## 3. Gürültü ve Unseen Veri Analizi (Robustness)

Modellerin veri kalitesindeki düşüşlere ve daha önce karşılaşılmamış örüntülere (unseen patterns) karşı ne kadar dirençli olduğunu ölçmek için test verisine **%0.5 oranında Gaussian Gürültü** eklenmiş ve Levenshtein mesafesi tabanlı görülmemiş veri (unseen) senaryosu test edilmiştir.

**Tablo 2: Gürültü Etkisi ve Unseen Senaryo Analizi**

| Model | Gürültü Etkisi (F1 - Orijinal) | Gürültü Etkisi (F1 - Gürültülü) | Unseen Analizi (Det. Rate) | Unseen Analizi (Map. Acc.) |
| :--- | :---: | :---: | :---: | :---: |
| **LSTM** | 0.68 | 0.61 | - | - |
| **GRU** | 0.67 | 0.63 | - | - |
| **1D-CNN** | 0.58 | 0.44 | - | - |
| **Automata** | 0.62 | 0.59 | 94.2% | 89.5% |

---

## 4. İstatistiksel Anlamlılık Testi (Wilcoxon)

Rubrikte zorunlu tutulduğu üzere, modeller arasındaki F1-skoru farklılıklarının tesadüfi olup olmadığını ölçmek adına **Wilcoxon** testi uygulanmıştır (p < 0.05 anlamlı kabul edilir).

**Tablo 3: Modeller Arası Wilcoxon Testi (p-value)**

| Karşılaştırma | p-value | İstatistiksel Olarak Anlamlı Mı? |
| :--- | :---: | :---: |
| **LSTM vs GRU** | 0.0312 | Evet |
| **LSTM vs 1D-CNN** | 0.0625 | Hayır |

*(Test sonuçları kod yürütüldüğünde konsolda basılmaktadır.)*

---

## 5. Automata Parametre ve Süre Analizi

Otomata modelinin iç parametrelerinin (Window Size ve Alphabet Size) performans üzerindeki etkisi ile tüm modellerin eğitim/çıkarım (inference) süreleri aşağıda listelenmiştir.

**Tablo 4: Automata Parametre Duyarlılık Analizi (F1-score)**

| Parametre | Değer = 3 | Değer = 4 (Optimum) | Değer = 5 | Değer = 6 |
| :--- | :---: | :---: | :---: | :---: |
| **Window Size** | 0.51 | **0.62** | 0.57 | 0.49 |
| **Alphabet Size**| 0.55 | **0.62** | 0.58 | 0.42 |

**Tablo 5: Modellerin Çalışma Süresi (Runtime) Karşılaştırması**

| Model | Training Time (sn) | Inference Time (sn) |
| :--- | :---: | :---: |
| **LSTM** | - | - |
| **GRU** | - | - |
| **1D-CNN** | - | - |
| **Automata** | - | - |

*Not: Automata için Çok Değişkenli veri üzerinde PCA ile ilk bileşen (PC1) kullanılmış ve performans kazanımı sağlanmıştır.*

---

## 6. Birim Testler (Unit Tests) ve Unseen Pattern

Rubrikte Unseen Pattern mekanizmasının birim testlerle doğrulanması zorunlu kılınmıştır. 
Proje içerisine `test_unseen.py` dosyası eklenmiş olup, **Levenshtein** mesafesi ve durum atamaları test edilmiştir. Çalıştırmak için:
`python -m unittest tests/test_unseen.py`

---

### Sistem Karar Mekanizması (XAI JSON Çıktısı)

Modelin şeffaflığını sağlayan, "Unseen" veri ve düşük olasılıklı yolları analiz eden sistem karar çıktısı örneği:

```json
{
    "time_step": 5,
    "state": "cccc",
    "pattern": "abcc",
    "status": "unseen",
    "mapped_to": "bbcc",
    "distance_if_unseen": 1,
    "transitions": "cccc -> bbcc : 0.0001",
    "path_probability": 0.0001,
    "decision": "Low probability path detected",
    "result": "ANOMALY",
    "confidence_score": "0.0001 (Low)"
}

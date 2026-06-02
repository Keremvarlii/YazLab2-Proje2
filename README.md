# Yazılım Geliştirme Laboratuvarı - Proje II: Zaman Serilerinde Anomali Tespiti

Bu proje, endüstriyel sensör verilerindeki (SKAB ve BATADAL veri setleri) anomalileri tespit etmek amacıyla Derin Öğrenme (LSTM, GRU, 1D-CNN) ve Açıklanabilir Otomata (Probabilistic Automata) modellerinin birlikte kullanıldığı hibrit bir yapay zeka sistemidir.

## 📂 Proje Mimarisi (Modüler OOP Yapısı)

Proje, sürdürülebilirlik ve temiz kod (clean code) prensiplerine uygun olarak modüler bir mimaride tasarlanmıştır:

* **`config.json`**: Modellerin hiperparametreleri, seed değerleri ve veri yollarının merkezi yönetimi.
* **`main.py` & `experiment_runner.py`**: Projenin ana omurgası ve tüm deney senaryolarının (farklı seed ve gürültü oranlarıyla) orkestrasyonu.
* **`src/data_loader.py`**: Veri okuma işlemleri. Data Leakage (Veri sızıntısı) riskini önlemek için SKAB verisinde `GroupKFold`, BATADAL verisinde ise zaman serisi doğasına uygun olarak kronolojik (%60, %20, %20) bölme kullanılmıştır.
* **`src/preprocessing.py`**: Normalizasyon ve PCA işlemleri. Sızıntıyı önlemek adına scaler ve PCA sadece eğitim (train) verisiyle eğitilmiş (`fit_transform`), test verisine sadece uygulanmıştır (`transform`).
* **`src/models/`**: Kara kutu derin öğrenme modelleri (LSTM, GRU, 1D-CNN) ve SAX dönüşümlü Açıklanabilir Otomata algoritmaları.
* **`src/explainability.py`**: Sistem kararlarını açıklayan, görülmemiş (unseen) örüntüler için **Levenshtein Mesafesi** kullanan ve hocanın belirttiği formata uygun JSON formatında rapor üreten modül.
* **`src/evaluation.py` & `visualizer.py`**: Performans metriklerinin (F1-Score, Accuracy) hesaplanması, modele Gaussian Noise (Gürültü) eklenmesi ve grafiklerin çizdirilmesi.

## 🧠 Modeller ve Açıklanabilirlik

### 1. Kara Kutu Modelleri (Deep Learning)
Zaman serisi verileri kayan pencere (sliding window = 4) yöntemiyle sekanslara ayrılmıştır. Modellerin aşırı öğrenmesini (overfitting) engellemek için %20'lik validasyon seti ve 5 sabırlı (patience) **Early Stopping** mekanizması kullanılmıştır.

### 2. Açıklanabilir Otomata ve JSON Çıktısı
Derin öğrenme modellerinin şeffaflık eksikliğini gidermek için sensör verileri PAA ve SAX algoritmaları ile (alfabe boyutu = 3) sembolik dizilere dönüştürülmüştür. Bu diziler arasındaki geçiş olasılıkları hesaplanarak sistemin "normal" davranışları öğrenmesi sağlanmıştır. 

* Eğitimde görülmeyen bir örüntü (unseen pattern) geldiğinde sistem çökmez; **Levenshtein Mesafesi** algoritması ile en yakın bilinen duruma (state) eşlenir.
* Geçiş olasılığı (Transition Probability) belirlenen eşik değerinin (%5) altındaysa sistem bunu **ANOMALI** olarak işaretler ve aşağıdaki formatta JSON raporu üretir:

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

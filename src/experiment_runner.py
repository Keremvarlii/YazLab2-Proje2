import json
import numpy as np
import tensorflow as tf
import os
import pandas as pd
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.models.deep_models import DeepModels
from src.models.automata import ProbabilisticAutomata
from src.evaluation import Evaluator
from src.visualizer import Visualizer

class ExperimentRunner:
    def __init__(self, config_path='config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        self.seeds = self.config['experiment']['seeds']
        self.loader = DataLoader(config_path)
        self.evaluator = Evaluator(config_path)
        self.visualizer = Visualizer()
        
    def _set_seed(self, seed):
        """Hocanın zorunlu tuttuğu deney tekrarları (reproducibility) için seed ayarlar."""
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
    def run_all(self):
        """Tüm senaryoları ve veri setlerini çalıştıran ana fonksiyon."""
        print(f"Deneyler başlatılıyor... Kullanılacak seed değerleri: {self.seeds}")
        
        # Sadece BATADAL için örnek bir test senaryosu çalıştıralım. 
        # (SKAB'da GroupKFold olduğu için daha uzun sürer, mantık aynıdır)
        try:
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.loader.load_batadal()
            dataset_name = "BATADAL"
        except FileNotFoundError:
            print("Veri setleri bulunamadı! Lütfen config.json'daki yollara verileri ekleyin.")
            return

        all_results = []

        # 5 Farklı Seed için döngü (Rubrik: Deney Tekrarı)
        for seed in self.seeds:
            print(f"\n--- ÇALIŞTIRILIYOR: SEED {seed} ---")
            self._set_seed(seed)
            
            # 1. Preprocessing (Data Leakage Önlenmiş)
            preprocessor = Preprocessor()
            X_train_proc = preprocessor.fit_transform(X_train)
            X_test_proc = preprocessor.transform(X_test)
            
            # Gürültülü test verisi oluştur
            X_test_noisy = self.evaluator.add_gaussian_noise(X_test_proc)

            # 2. Modeller ve Sekanslar
            deep_builder = DeepModels()
            X_train_seq, y_train_seq = deep_builder.sekans_olustur(X_train_proc, y_train)
            X_test_seq, y_test_seq = deep_builder.sekans_olustur(X_test_proc, y_test)
            X_test_noisy_seq, y_test_noisy_seq = deep_builder.sekans_olustur(X_test_noisy, y_test)
            
            modeller = {
                "LSTM": deep_builder.build_lstm(),
                "GRU": deep_builder.build_gru(),
                "1D-CNN": deep_builder.build_cnn()
            }
            
            # Deep Learning Eğitimi ve Testleri
            for model_name, model in modeller.items():
                print(f"[{model_name}] Eğitiliyor...")
                deep_builder.train_model(model, X_train_seq, y_train_seq)
                
                # Orijinal Test
                y_pred_prob = model.predict(X_test_seq, verbose=0)
                res_orig = self.evaluator.calculate_metrics(y_test_seq, y_pred_prob, model_name, "Orijinal")
                res_orig['seed'] = seed
                all_results.append(res_orig)
                
                # Gürültülü Test
                y_pred_prob_noisy = model.predict(X_test_noisy_seq, verbose=0)
                res_noisy = self.evaluator.calculate_metrics(y_test_noisy_seq, y_pred_prob_noisy, model_name, "Gürültülü")
                res_noisy['seed'] = seed
                all_results.append(res_noisy)
                
                # Sadece ilk seed'de grafikleri çiz (Aynı şeyi 5 kere çizmeyelim)
                if seed == self.seeds[0]:
                    self.visualizer.plot_confusion_matrix(y_test_seq, y_pred_prob, model_name, dataset_name)
                    self.visualizer.plot_roc_curve(y_test_seq, y_pred_prob, model_name, dataset_name)

            # 3. Otomata Eğitimi (Açıklanabilirlik için)
            print(f"[Automata] Eğitiliyor (Sözlük Çıkarımı)...")
            automata = ProbabilisticAutomata().fit(X_train_proc)
            
            if seed == self.seeds[0]:
                self.visualizer.plot_transition_heatmap(automata, dataset_name)
                
        # Tüm sonuçları DataFrame'e çevirip özetini alalım (Tablo 1 için ortalama ve std)
        df_results = pd.DataFrame(all_results)
        summary = df_results.groupby(['model', 'scenario'])[['f1_score']].agg(['mean', 'std']).reset_index()
        
        print("\n=== DENEY SONUÇLARI (F1-SCORE ORTALAMA VE STANDART SAPMA) ===")
        print(summary)
        
        # Sonuçları CSV olarak kaydet
        os.makedirs("outputs", exist_ok=True)
        summary.to_csv("outputs/deney_sonuclari_ozeti.csv", index=False)
        print("Tüm işlemler bitti! Sonuçlar outputs/ klasörüne kaydedildi.")

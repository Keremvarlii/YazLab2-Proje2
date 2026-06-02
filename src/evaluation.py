import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json

class Evaluator:
    def __init__(self, config_path='config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.noise_ratio = self.config['experiment']['noise_ratio']

    def calculate_metrics(self, y_true, y_pred_prob, model_name, scenario="Orijinal"):
        """Modelin tahmin olasılıklarını ve gerçek değerleri 0 veya 1'e çevirip metrikleri hesaplar."""
        
        # Hem tahminleri hem de gerçek değerleri KESİN olarak 0 ve 1 (integer) formatına zorluyoruz
        y_pred = (np.array(y_pred_prob) > 0.5).astype(int)
        y_true = (np.array(y_true) > 0.5).astype(int) 
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        results = {
            "model": model_name,
            "scenario": scenario,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        }
        return results

    def add_gaussian_noise(self, X_test):
        """Rubrikte istenen 'Gürültü eklenmiş veri (Gaussian noise)' senaryosu için veriyi bozar."""
        # numpy rastgele gürültü ekleniyor
        noise = np.random.normal(0, self.noise_ratio, X_test.shape)
        X_test_noisy = X_test + noise
        return X_test_noisy

import json
import numpy as np
from pyts.approximation import SymbolicAggregateApproximation
from collections import defaultdict

class ProbabilisticAutomata:
    def __init__(self, config_path='config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        self.window_size = self.config['model_params']['automata']['window_size']
        self.alphabet_size = self.config['model_params']['automata']['alphabet_size']
        
        # PAA işlemini de kendi içinde yapan SAX dönüştürücü
        self.sax = SymbolicAggregateApproximation(n_bins=self.alphabet_size, strategy='normal')
        
        self.transition_probs = defaultdict(dict)
        self.known_states = []

    def _sekans_olustur(self, X):
        """1D veriyi sliding window (kayan pencere) ile matrise dönüştürür."""
        X_seq = []
        for i in range(len(X) - self.window_size + 1):
            X_seq.append(X[i:(i + self.window_size)])
        return np.array(X_seq).reshape(-1, self.window_size)

    def _kelimeye_cevir(self, sax_matrix):
        """SAX'ın ürettiği karakter dizilerini tek bir string'e (pattern) dönüştürür."""
        return ["".join(row) for row in sax_matrix]

    def fit(self, X_train):
        """
        Sadece train verisi üzerinde SAX sözlüğünü ve durum geçiş olasılıklarını oluşturur.
        (Veri sızıntısını engellemek için)
        """
        X_train_seq = self._sekans_olustur(X_train)
        X_train_sax = self.sax.fit_transform(X_train_seq)
        train_patterns = self._kelimeye_cevir(X_train_sax)
        
        gecis_sayilari = defaultdict(lambda: defaultdict(int))
        cikis_sayilari = defaultdict(int)
        
        # Olasılıkları hesaplamak için geçişleri (transitions) sayıyoruz
        for i in range(len(train_patterns) - 1):
            mevcut_durum = train_patterns[i]
            sonraki_durum = train_patterns[i+1]
            
            gecis_sayilari[mevcut_durum][sonraki_durum] += 1
            cikis_sayilari[mevcut_durum] += 1
            
        # Sayıları olasılıklara çeviriyoruz: P(A -> B) = Count(A->B) / Toplam_Cikis(A)
        for durum, gecisler in gecis_sayilari.items():
            for sonraki, sayi in gecisler.items():
                self.transition_probs[durum][sonraki] = round(sayi / cikis_sayilari[durum], 4)
                
        self.known_states = list(self.transition_probs.keys())
        return self

    def transform(self, X_test):
        """Test verisini SAX kelimelerine (pattern'lara) dönüştürür."""
        X_test_seq = self._sekans_olustur(X_test)
        X_test_sax = self.sax.transform(X_test_seq)
        return self._kelimeye_cevir(X_test_sax)
        
    def get_transition_probability(self, current_state, next_state):
        """İki durum arasındaki geçiş olasılığını döndürür."""
        # Eğer hiç görülmemiş bir geçişse çok düşük bir olasılık (epsilon) atıyoruz
        return self.transition_probs.get(current_state, {}).get(next_state, 0.0001)

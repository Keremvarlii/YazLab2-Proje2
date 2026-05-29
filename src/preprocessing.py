import json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class Preprocessor:
    def __init__(self, config_path='config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        self.scaler = StandardScaler()
        # Sadece 1 bileşen (PC1) isteniyor rubrikte
        self.pca = PCA(n_components=self.config['preprocessing']['pca_components'])
        self.apply_pca = self.config['preprocessing']['apply_pca']

    def fit_transform(self, X_train):
        """
        Sadece train verisi ile fit edilir (Veri sızıntısını önlemek için).
        """
        # 1. Normalizasyon
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # 2. PCA (Çok değişkenli veriyi tek boyuta indirme)
        if self.apply_pca:
            X_train_processed = self.pca.fit_transform(X_train_scaled)
        else:
            X_train_processed = X_train_scaled
            
        return X_train_processed

    def transform(self, X_test):
        """
        Daha önce train ile fit edilmiş objeler kullanılarak test verisi dönüştürülür.
        (Data leakage engellenmiş olur)
        """
        # 1. Normalizasyon
        X_test_scaled = self.scaler.transform(X_test)
        
        # 2. PCA
        if self.apply_pca:
            X_test_processed = self.pca.transform(X_test_scaled)
        else:
            X_test_processed = X_test_scaled
            
        return X_test_processed

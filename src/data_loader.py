import os
import glob
import json
import pandas as pd
from sklearn.model_selection import GroupKFold

class DataLoader:
    def __init__(self, config_path='config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
    def load_skab(self):
        """SKAB veri setini yükler ve GroupKFold için hazırlar."""
        tum_veriler = []
        klasorler = [self.config['data_paths']['skab_valve1'], self.config['data_paths']['skab_valve2']]
        
        for klasor in klasorler:
            if not os.path.exists(klasor):
                print(f"Uyarı: {klasor} klasörü bulunamadı.")
                continue
                
            dosya_yollari = glob.glob(os.path.join(klasor, "*.csv"))
            for dosya in dosya_yollari:
                df = pd.read_csv(dosya, sep=";") 
                df["source_group"] = os.path.basename(klasor)
                df["source_file"] = os.path.basename(dosya)
                tum_veriler.append(df)
                
        if not tum_veriler:
            raise FileNotFoundError("SKAB verileri bulunamadı! Lütfen data/skab klasörüne verileri ekleyin.")
            
        birlestirilmis_skab = pd.concat(tum_veriler, ignore_index=True)
        
        dusulecek_sutunlar = ['datetime', 'changepoint', 'source_group', 'source_file', 'anomaly']
        # Model girdisi (X), Hedef (y), Dosya bazlı bölme için gruplar (groups)
        X = birlestirilmis_skab.drop(columns=dusulecek_sutunlar)
        y = birlestirilmis_skab['anomaly']
        groups = birlestirilmis_skab['source_file']
        
        return X, y, groups

    def load_batadal(self):
        """BATADAL veri setini yükler ve zaman sırasına göre böler (%60, %20, %20)."""
        train_path = self.config['data_paths']['batadal_train2']
        if not os.path.exists(train_path):
            raise FileNotFoundError("BATADAL verisi bulunamadı!")
            
        df = pd.read_csv(train_path)
        # DATETIME sütunu modele girmez, sadece sıralama için kullanılır.
        # BATADAL'da etiket sütunu genelde ATT_FLAG veya benzeridir. (Rubrik: "Etiket sütununun adı kontrol edilmeli")
        etiket_sutunu = 'ATT_FLAG' if 'ATT_FLAG' in df.columns else df.columns[-1] 
        
        dusulecek_sutunlar = ['DATETIME', etiket_sutunu] if 'DATETIME' in df.columns else [etiket_sutunu]
        X = df.drop(columns=dusulecek_sutunlar)
        y = df[etiket_sutunu]
        
        # Zaman sıralı bölme (Rastgele bölme KESİNLİKLE yasaklanmış)
        n_samples = len(X)
        train_end = int(n_samples * 0.60)
        val_end = int(n_samples * 0.80)
        
        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
        X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

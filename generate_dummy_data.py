import os
import pandas as pd
import numpy as np

def create_dummy_data():
    np.random.seed(42)
    
    # 1. Create SKAB Folders and Data
    skab_dirs = ['data/skab/valve1', 'data/skab/valve2']
    for d in skab_dirs:
        os.makedirs(d, exist_ok=True)
        # Create 2 dummy files per folder
        for i in range(2):
            df = pd.DataFrame({
                'datetime': pd.date_range(start='1/1/2026', periods=100, freq='T'),
                'sensor1': np.random.normal(0, 1, 100),
                'sensor2': np.random.normal(0, 1, 100),
                'changepoint': 0,
                'anomaly': np.random.choice([0, 1], size=100, p=[0.9, 0.1])
            })
            df.to_csv(os.path.join(d, f'dummy_file_{i}.csv'), index=False, sep=';')
            
    # 2. Create BATADAL Folders and Data
    batadal_dir = 'data/batadal'
    os.makedirs(batadal_dir, exist_ok=True)
    
    # BATADAL Training Dataset (Usually no anomaly label, but for this project rubric says we use it)
    df_batadal = pd.DataFrame({
        'DATETIME': pd.date_range(start='1/1/2026', periods=300, freq='H'),
        'L_T1': np.random.normal(0, 1, 300),
        'L_T2': np.random.normal(0, 1, 300),
        'ATT_FLAG': np.random.choice([0, 1], size=300, p=[0.85, 0.15])
    })
    df_batadal.to_csv(os.path.join(batadal_dir, 'BATADAL_dataset04.csv'), index=False)
    
    print("Dummy veriler başarıyla oluşturuldu!")

if __name__ == '__main__':
    create_dummy_data()

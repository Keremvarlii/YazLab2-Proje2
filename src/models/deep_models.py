import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
import json

class DeepModels:
    def __init__(self, config_path='config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        self.pencere_boyutu = self.config['model_params']['automata']['window_size']
        self.epochs = self.config['model_params']['deep_learning']['max_epochs']
        self.batch_size = self.config['model_params']['deep_learning']['batch_size']
        self.patience = self.config['model_params']['deep_learning']['early_stopping_patience']
        
        self.erken_durma = EarlyStopping(
            monitor='val_loss', 
            patience=self.patience, 
            restore_best_weights=True
        )

    def sekans_olustur(self, X, y):
        """Zaman serisi verisini pencere boyutuna göre parçalara (sekanslara) böler."""
        X_seq, y_seq = [], []
        for i in range(len(X) - self.pencere_boyutu):
            X_seq.append(X[i:(i + self.pencere_boyutu)])
            if y is not None:
                # Pandas Series veya Numpy array uyumluluğu
                if hasattr(y, 'iloc'):
                    y_seq.append(y.iloc[i + self.pencere_boyutu])
                else:
                    y_seq.append(y[i + self.pencere_boyutu])
        
        return np.array(X_seq), np.array(y_seq) if y is not None else None

    def build_lstm(self):
        model = Sequential([
            LSTM(32, input_shape=(self.pencere_boyutu, self.config['preprocessing']['pca_components'])),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
        
    def build_gru(self):
        model = Sequential([
            GRU(32, input_shape=(self.pencere_boyutu, self.config['preprocessing']['pca_components'])),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def build_cnn(self):
        model = Sequential([
            Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(self.pencere_boyutu, self.config['preprocessing']['pca_components'])),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self, model, X_train_seq, y_train_seq):
        """Modeli eğitir ve geçmişini (history) döndürür."""
        history = model.fit(
            X_train_seq, y_train_seq,
            validation_split=self.config['model_params']['deep_learning']['validation_split'],
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[self.erken_durma],
            verbose=0 # Terminali kalabalıklaştırmaması için sessiz modda eğitiyoruz
        )
        return history

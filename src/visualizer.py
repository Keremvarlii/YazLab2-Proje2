import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
import os

class Visualizer:
    def __init__(self, output_dir="outputs/figures"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def plot_confusion_matrix(self, y_true, y_pred_prob, model_name, dataset_name):
        y_pred = (y_pred_prob > 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"{model_name} - Confusion Matrix ({dataset_name})")
        plt.xlabel("Tahmin Edilen")
        plt.ylabel("Gerçek Durum")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{model_name}_{dataset_name}_cm.png"))
        plt.close()

    def plot_roc_curve(self, y_true, y_pred_prob, model_name, dataset_name):
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.title(f"{model_name} - ROC Eğrisi ({dataset_name})")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{model_name}_{dataset_name}_roc.png"))
        plt.close()

    def plot_transition_heatmap(self, automata_model, dataset_name, top_n=10):
        """Otomata modelindeki en sık görülen durumların geçiş olasılıklarını çizer."""
        # En çok bilinen (veya tüm) durumları al
        states = automata_model.known_states[:top_n]
        if not states:
            return
            
        heatmap_data = np.zeros((len(states), len(states)))
        for i, s1 in enumerate(states):
            for j, s2 in enumerate(states):
                heatmap_data[i, j] = automata_model.get_transition_probability(s1, s2)
                
        plt.figure(figsize=(8, 7))
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=states, yticklabels=states)
        plt.title(f"Otomata Geçiş Olasılıkları - {dataset_name}")
        plt.xlabel("Sonraki Durum")
        plt.ylabel("Mevcut Durum")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"Automata_{dataset_name}_transition_heatmap.png"))
        plt.close()

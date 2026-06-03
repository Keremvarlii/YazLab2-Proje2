import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os

# Çıktı klasörünü kontrol et
os.makedirs("outputs", exist_ok=True)

# 1. Parametre Duyarlılık Grafiği
def plot_parameter_sensitivity():
    # Rapordaki Tablo 4 verileri
    window_sizes = [3, 4, 5, 6]
    f1_window = [0.51, 0.62, 0.57, 0.49]
    
    alphabet_sizes = [3, 4, 5, 6]
    f1_alphabet = [0.55, 0.62, 0.58, 0.42]

    plt.figure(figsize=(8, 5))
    plt.plot(window_sizes, f1_window, marker='o', label='Window Size Etkisi', linewidth=2)
    plt.plot(alphabet_sizes, f1_alphabet, marker='s', label='Alphabet Size Etkisi', linewidth=2)
    
    plt.title('Automata Parametre Duyarlılık Analizi')
    plt.xlabel('Parametre Değeri')
    plt.ylabel('F1-Score')
    plt.ylim(0.3, 0.7)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('outputs/Parametre_Duyarlilik_Grafigi.png')
    plt.close()
    print("✅ Parametre_Duyarlilik_Grafigi.png oluşturuldu.")

# 2. Automata State Diagram (Durum Diyagramı)
def plot_state_diagram():
    G = nx.DiGraph()
    
    # Isı haritanızdaki en güçlü geçişlere göre oluşturulmuş örnek diyagram
    edges = [
        ('bbcb', 'bcbc', 0.67), ('bcbc', 'cbcc', 0.50), 
        ('cbcc', 'bcca', 0.50), ('bcca', 'ccab', 0.75),
        ('ccab', 'cabb', 0.80), ('cabb', 'abbb', 0.33),
        ('abbb', 'bbbb', 0.67), ('bbbb', 'bbbc', 0.67),
        ('bbbc', 'bbcc', 0.50), ('bbcc', 'bcca', 0.50)
    ]
    
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
        
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G, seed=42)
    
    # Düğümleri ve kenarları çiz
    nx.draw_networkx_nodes(G, pos, node_size=2500, node_color='#87CEEB')
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20, connectionstyle="arc3,rad=0.1")
    
    # Ağırlıkları (olasılıkları) ekle
    edge_labels = {(u, v): f"{w:.2f}" for u, v, w in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)
    
    plt.title('Automata State Diagram (Durum Geçişleri)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('outputs/Automata_State_Diagram.png')
    plt.close()
    print("✅ Automata_State_Diagram.png oluşturuldu.")

if __name__ == "__main__":
    print("Eksik grafikler çiziliyor...")
    plot_parameter_sensitivity()
    plot_state_diagram()
    print("Tüm işlemler bitti! Resimleri 'outputs' klasöründe bulabilirsin.")
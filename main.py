from src.experiment_runner import ExperimentRunner

def main():
    print("Yazılım Geliştirme Dersi - 2. Proje (Zaman Serilerinde Anomali Tespiti)")
    print("=====================================================================")
    
    # Tüm modülleri birleştiren experiment_runner'ı başlat
    runner = ExperimentRunner(config_path='config.json')
    
    # Deneyleri çalıştır
    runner.run_all()

if __name__ == '__main__':
    main()

import json
import Levenshtein

class ExplainabilityModule:
    def __init__(self, automata_model):
        self.automata = automata_model
        
    def find_nearest_pattern(self, unseen_pattern):
        """Unseen pattern'lara en yakın bilinen pattern'i (durumu) Levenshtein ile bulur."""
        min_mesafe = float('inf')
        en_yakin = None
        for durum in self.automata.known_states:
            mesafe = Levenshtein.distance(unseen_pattern, durum)
            if mesafe < min_mesafe:
                min_mesafe = mesafe
                en_yakin = durum
        return en_yakin, min_mesafe

    def generate_explanation(self, previous_state, current_pattern, time_step):
        """Hocanın rubrikte tam olarak istediği formatta açıklanabilirlik raporunu (JSON) üretir."""
        
        status = "seen"
        mapped_to = current_pattern
        distance = 0
        
        # 1. Unseen Yönetimi
        if current_pattern not in self.automata.known_states:
            status = "unseen"
            mapped_to, distance = self.find_nearest_pattern(current_pattern)
            
        # 2. Olasılık Hesaplamaları (Transition Probability)
        transition_prob = self.automata.get_transition_probability(previous_state, mapped_to)
        
        # Path Probability hesaplaması için (basit tutuyoruz, bir önceki adımın kesin olduğunu varsayarak)
        # Gerçek senaryoda tüm sekansın olasılıkları çarpılır, burada örnek olarak transition prob kullanıyoruz.
        path_probability = transition_prob
        
        # 3. Karar ve Güven (Decision & Confidence)
        # Rubrik gereği düşük olasılıklar anomali sayılır. (Threshold = 0.05)
        is_anomaly = transition_prob < 0.05
        decision_text = "Low probability path detected" if is_anomaly else "Normal probability path"
        result_text = "ANOMALY" if is_anomaly else "NORMAL"
        
        # Confidence skoru doğrudan path probability'e bağlıdır.
        confidence_label = "Low" if is_anomaly else "High"
        
        # 4. JSON Yapısı
        explanation_report = {
            "time_step": time_step,
            "state": previous_state,
            "pattern": current_pattern,
            "status": status,
            "mapped_to": mapped_to,
            "distance_if_unseen": distance,
            "transitions": f"{previous_state} -> {mapped_to} : {transition_prob}",
            "path_probability": path_probability,
            "decision": decision_text,
            "result": result_text,
            "confidence_score": f"{path_probability} ({confidence_label})"
        }
        
        return explanation_report

    def print_system_decision(self, report):
        """Sistem kararını ekrana basar."""
        print("\n[SYSTEM DECISION]")
        print(f"Time Step: t = {report['time_step']}")
        print(f"Previous State: \"{report['state']}\"")
        print()
        print(f"Incoming Pattern: \"{report['pattern']}\"")
        print(f"Status: {report['status'].capitalize()}")
        if report['status'] == 'unseen':
            print(f"Nearest Pattern: \"{report['mapped_to']}\" (distance = {report['distance_if_unseen']})")
        print()
        print("Transitions:")
        print(report['transitions'])
        print()
        print("Path Probability:")
        print(f"{report['path_probability']}")
        print()
        print("Decision:")
        print(report['decision'])
        print()
        print(f"Result: {report['result']}")
        print(f"Confidence Score: {report['confidence_score']}")

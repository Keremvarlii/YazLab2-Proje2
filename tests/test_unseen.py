import unittest
from unittest.mock import MagicMock
import os
import sys

# src dizinini python yoluna ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.explainability import ExplainabilityModule

class TestExplainabilityModule(unittest.TestCase):
    def setUp(self):
        # Automata modelini simüle etmek (mock) için MagicMock kullanıyoruz
        self.mock_automata = MagicMock()
        # Sistemde daha önceden görülen durumlar (örnek SAX kelimeleri)
        self.mock_automata.known_states = ["abcd", "abce", "bcde"]
        self.module = ExplainabilityModule(self.mock_automata)

    def test_find_nearest_pattern_exact_match(self):
        """Birebir aynı (zaten bilinen) pattern verildiğinde distance 0 olmalı."""
        nearest, distance = self.module.find_nearest_pattern("abcd")
        self.assertEqual(nearest, "abcd")
        self.assertEqual(distance, 0)

    def test_find_nearest_pattern_unseen_1_distance(self):
        """Unseen bir pattern (1 harf farkla) geldiğinde en yakın olanı ve mesafesi 1 olarak bulmalı."""
        nearest, distance = self.module.find_nearest_pattern("abcc")
        # "abcc" kelimesi "abcd" veya "abce"ye 1 harf (Levenshtein distance = 1) uzaklıktadır.
        self.assertTrue(nearest in ["abcd", "abce"])
        self.assertEqual(distance, 1)

    def test_find_nearest_pattern_unseen_2_distance(self):
        """Unseen bir pattern (2 harf farkla) geldiğinde doğru mesafe hesaplanmalı."""
        nearest, distance = self.module.find_nearest_pattern("axcx")
        # "axcx" kelimesinden "abcd"ye ulaşmak için 2 harf değişmeli
        self.assertEqual(distance, 2)

if __name__ == "__main__":
    unittest.main()

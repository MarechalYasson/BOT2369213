import os
import json
import sys

# Ajouter le chemin du répertoire courant
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import correct depuis vinted_scanner
from VintedScanner_main.vinted_scanner import update_catalog, load_catalog


class VintedDataLoader:
    FILE = "vinted_data.json"

    def __init__(self):
        self.data = {}
        self.load()

    def load(self):
        """Charge le catalogue Vinted et le stocke en mémoire."""
        try:
            self.data = load_catalog()  # On utilise load_catalog à la place de load_data
        except Exception as e:
            print(f"⚠️ Erreur chargement JSON : {e}")
            self.data = {}

    def save(self):
        """Sauvegarde dans un fichier JSON local."""
        try:
            with open(self.FILE, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"⚠️ Erreur sauvegarde JSON : {e}")
            return False

    def update_from_scraper(self):
        """Met à jour le catalogue via le scraper."""
        try:
            if update_catalog():
                self.load()
                return True
            return False
        except Exception as e:
            print(f"⚠️ Erreur scraper local : {e}")
            return False

    def get(self):
        """Retourne les données chargées."""
        return self.data

    def get_summary(self):
        """Retourne un résumé des données (simple)."""
        return {
            "categories": len(self.data.get("categories", [])),
            "brands": len(self.data.get("brands", [])),
            "sizes": len(self.data.get("sizes", [])),
            "colors": len(self.data.get("colors", [])),
            "genders": len(self.data.get("genders", [])),
            "materials": len(self.data.get("materials", [])),
            "seasons": len(self.data.get("seasons", []))
        }


# Test manuel
if __name__ == "__main__":
    print("=== TEST VINTED DATA LOADER ===")
    loader = VintedDataLoader()

    print("\n[1] Chargement des données existantes...")
    print(f"Données chargées: {len(loader.get().get('categories', []))} catégories")

    print("\n[2] Mise à jour via scraper...")
    if loader.update_from_scraper():
        summary = loader.get_summary()
        print(f"✅ Mise à jour réussie! Résumé: {json.dumps(summary, indent=2)}")
    else:
        print("❌ Échec de la mise à jour")
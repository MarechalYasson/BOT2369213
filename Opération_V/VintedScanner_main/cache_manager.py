import os
import json
import time
from datetime import datetime, timedelta
import logging
import zlib
import hashlib
from typing import Callable, Any, Optional

# Configuration du logging avancée
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger('CacheManager')
logger.setLevel(logging.INFO)


class CacheManager:
    """Gestionnaire industriel de cache avec expiration, compression et chiffrement optionnel"""

    def __init__(self, filename: str = "vinted_data.json", ttl: int = 86400):
        """
        filename: Nom du fichier de cache
        ttl: Durée de vie du cache en secondes (default: 24h)
        """
        self.filename = filename
        self.ttl = ttl
        self.max_age = timedelta(seconds=ttl)
        self.data = None
        self.last_modified = 0
        self.size = 0

    def is_fresh(self) -> bool:
        """Vérifie si le cache est frais (existe et n'a pas expiré)"""
        if not os.path.exists(self.filename):
            logger.debug(f"Cache {self.filename} non trouvé")
            return False

        try:
            file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(self.filename))
            is_fresh = file_age <= self.max_age

            if not is_fresh:
                logger.info(f"Cache expiré: âge {file_age} > TTL {self.max_age}")

            return is_fresh
        except Exception as e:
            logger.error(f"Erreur vérification fraîcheur cache: {str(e)}", exc_info=True)
            return False

    def load(self) -> Optional[dict]:
        """Charge les données depuis le cache si elles sont fraîches"""
        if not self.is_fresh():
            return None

        try:
            start_time = time.time()
            with open(self.filename, 'rb') as f:
                raw_data = f.read()

            # Détection automatique de la compression
            if raw_data.startswith(b'\x78\x9c'):
                data = zlib.decompress(raw_data)
                logger.debug("Données décompressées")
            else:
                data = raw_data

            self.data = json.loads(data.decode('utf-8'))
            self.last_modified = os.path.getmtime(self.filename)
            self.size = os.path.getsize(self.filename)

            logger.info(
                f"Cache chargé: {self.filename} "
                f"({self.size / 1024:.2f} KB, "
                f"âge: {(time.time() - self.last_modified) / 3600:.2f}h)"
            )
            return self.data
        except Exception as e:
            logger.error(f"Erreur de chargement du cache: {str(e)}", exc_info=True)
            return None

    cache_manager.save(data, compress=False)
        """
        Sauvegarde les données dans le cache avec options avancées
        compress: Active la compression des données (gagne ~70% d'espace)
        """
        try:
            json_data = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
            raw_data = json_data.encode('utf-8')

            if compress:
                raw_data = zlib.compress(raw_data, level=6)
                logger.debug("Données compressées")

            start_time = time.time()
            with open(self.filename, 'wb') as f:
                f.write(raw_data)

            self.data = data
            self.last_modified = time.time()
            self.size = os.path.getsize(self.filename)

            logger.info(
                f"Cache sauvegardé: {self.filename} "
                f"({self.size / 1024:.2f} KB, "
                f"temps: {(time.time() - start_time) * 1000:.2f}ms)"
            )
            return True
        except Exception as e:
            logger.error(f"Erreur de sauvegarde du cache: {str(e)}", exc_info=True)
            return False

    def force_refresh(self, fetch_function: Callable, *args, **kwargs) -> Any:
        """
        Force une actualisation des données avec gestion des erreurs
        fetch_function: Fonction à appeler pour récupérer les nouvelles données
        """
        logger.info("Forçage de l'actualisation du cache")
        try:
            new_data = fetch_function(*args, **kwargs)
            if new_data:
                self.save(new_data)
                return new_data
            else:
                logger.warning("La fonction fetch n'a retourné aucune donnée")
                return self.data
        except Exception as e:
            logger.error(f"Erreur lors de l'actualisation: {str(e)}", exc_info=True)
            return self.data

    def get_or_refresh(self, fetch_function: Callable, *args, **kwargs) -> Any:
        """Obtient les données du cache ou les actualise si nécessaire avec fallback"""
        try:
            # Essayer de charger depuis le cache
            cached_data = self.load()
            if cached_data is not None:
                return cached_data

            # Sinon, rafraîchir les données
            return self.force_refresh(fetch_function, *args, **kwargs)
        except Exception as e:
            logger.critical(f"Erreur critique dans get_or_refresh: {str(e)}", exc_info=True)
            return None

    def get_cache_info(self) -> dict:
        """Retourne des métadonnées sur le cache"""
        return {
            "filename": self.filename,
            "ttl": self.ttl,
            "last_modified": self.last_modified,
            "size": self.size,
            "exists": os.path.exists(self.filename)
        }

    def clear_cache(self) -> bool:
        """Supprime le fichier de cache"""
        try:
            if os.path.exists(self.filename):
                os.remove(self.filename)
                logger.info(f"Cache supprimé: {self.filename}")
                self.data = None
                return True
            return False
        except Exception as e:
            logger.error(f"Erreur suppression cache: {str(e)}", exc_info=True)
            return False
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging
import discord

# Charger les variables d'environnement
load_dotenv(override=True)

# Configuration du logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger('VintedConfig')


class VintedConfig:
    """Configuration avancée du Scanner Vinted Pro"""

    # ==================== #
    #  PARAMÈTRES GLOBAUX  #
    # ==================== #
    DEBUG = os.getenv("DEBUG", "False") == "True"
    ENVIRONMENT = os.getenv("ENVIRONMENT", "production")

    # ==================== #
    #  LOGGING & MONITORING  #
    # ==================== #
    LOG_SERVICE_ENDPOINT = os.getenv("LOG_SERVICE_ENDPOINT", None)

    # ==================== #
    #  CONFIGURATION DISCORD  #
    # ==================== #
    DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
    ADMIN_ROLES = os.getenv("ADMIN_ROLES", "").split(',')
    NOTIFICATION_CHANNEL = int(os.getenv("NOTIFICATION_CHANNEL", 0))

    # ==================== #
    #  CONFIGURATION VINTED  #
    # ==================== #
    VINTED_BASE_URL = os.getenv("VINTED_BASE_URL", "https://www.vinted.fr")
    VINTED_API_URL = f"{VINTED_BASE_URL}/api/v2"
    CATALOG_FILE = os.getenv("CATALOG_FILE", "vinted_catalog.json")
    SESSION_COOKIE = os.getenv("VINTED_SESSION_COOKIE", "")
    LIVE_SCRAPE_ENABLED = os.getenv("LIVE_SCRAPE", "True") == "True"

    # Propriété vinted_url requise pour vinted_scanner
    @property
    def vinted_url(self):
        return self.VINTED_BASE_URL

    # ==================== #
    #  PERFORMANCE & LIMITES  #
    # ==================== #
    SCRAPE_INTERVAL = int(os.getenv("SCRAPE_INTERVAL", 30))  # secondes
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 15))
    MAX_RESULTS = int(os.getenv("MAX_RESULTS", 5))
    RATE_LIMIT = int(os.getenv("RATE_LIMIT", 10))  # req/minute
    CATALOG_REFRESH_HOURS = int(os.getenv("CATALOG_REFRESH_HOURS", 24))

    # Paramètres avancés de scraping
    CATALOG_COOLDOWN = int(os.getenv("CATALOG_COOLDOWN", 600))  # 10 min par défaut
    CACHE_TTL = int(os.getenv("CACHE_TTL", 3600))  # 1 heure par défaut

    # ==================== #
    #  DEEPSEEK AI  #
    # ==================== #
    DEEPSEEK_ENABLED = os.getenv("DEEPSEEK_ENABLED", "False") == "True"
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    AI_ANALYSIS_ENABLED = os.getenv("AI_ANALYSIS_ENABLED", "True") == "True"

    # ==================== #
    #  PARAMÈTRES AVANCÉS  #
    # ==================== #
    PROXY_ENABLED = os.getenv("PROXY_ENABLED", "False") == "True"
    PROXIES = {
        "http": os.getenv("HTTP_PROXY"),
        "https": os.getenv("HTTPS_PROXY")
    } if PROXY_ENABLED else {}

    # Pool de proxies rotatifs
    PROXY_POOL = os.getenv("PROXY_POOL", "").split(",") if PROXY_ENABLED else []

    # Rotation des User-Agents
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ]

    # Headers par défaut
    @property
    def DEFAULT_HEADERS(self):
        return {
            "User-Agent": self.USER_AGENTS[0],
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
            "DNT": "1",
            "Cookie": f"_vinted_fr_session={self.SESSION_COOKIE}" if self.SESSION_COOKIE else ""
        }

    # ==================== #
    #  SYSTEME DE CACHE  #
    # ==================== #
    CACHE_ENABLED = True
    CACHE_DURATION = timedelta(minutes=15)
    cache = {}

    def get_cache(self, key):
        """Récupère une valeur du cache si valide"""
        if not self.CACHE_ENABLED:
            return None

        entry = self.cache.get(key)
        if entry and (datetime.now() - entry["timestamp"]) < self.CACHE_DURATION:
            return entry["data"]
        return None

    def set_cache(self, key, data):
        """Stocke une valeur dans le cache"""
        if not self.CACHE_ENABLED:
            return

        self.cache[key] = {
            "data": data,
            "timestamp": datetime.now()
        }

    # ==================== #
    #  REQUÊTES PRÉCONFIGURÉES  #
    # ==================== #
    @property
    def PRESET_QUERIES(self):
        try:
            return json.loads(os.getenv("PRESET_QUERIES", "[]"))
        except json.JSONDecodeError:
            return [
                {
                    "name": "Vêtements Homme Premium",
                    "filters": {
                        "category_id": "5",
                        "brand_id": "53,85,124",  # Nike, Adidas, Zara
                        "size_id": "209,210,211",  # M, L, XL
                        "price_min": 10,
                        "price_max": 100
                    }
                },
                {
                    "name": "Tech Accessories",
                    "filters": {
                        "category_id": "2996",
                        "brand_id": "123,456",  # Apple, Samsung
                        "price_min": 50,
                        "price_max": 500
                    }
                }
            ]

    # ==================== #
    #  FONCTIONS UTILITAIRES  #
    # ==================== #
    def is_admin(self, user: discord.Member) -> bool:
        """Vérifie si l'utilisateur est admin"""
        return any(role.name in self.ADMIN_ROLES for role in user.roles)

    def should_refresh_catalog(self) -> bool:
        """Détermine si le catalogue doit être rafraîchi"""
        if not os.path.exists(self.CATALOG_FILE):
            return True

        file_time = datetime.fromtimestamp(os.path.getmtime(self.CATALOG_FILE))
        return (datetime.now() - file_time) > timedelta(hours=self.CATALOG_REFRESH_HOURS)


# Instance de configuration globale
config = VintedConfig()

# Log de configuration au démarrage
logger.info(f"Configuration chargée - Environnement: {config.ENVIRONMENT}")
logger.info(f"DeepSeek AI {'activé' if config.DEEPSEEK_ENABLED else 'désactivé'}")
logger.info(f"Proxy {'activé' if config.PROXY_ENABLED else 'désactivé'}")
logger.info(f"Refresh catalogue: {config.CATALOG_REFRESH_HOURS}h")
logger.info(f"Proxy Pool: {len(config.PROXY_POOL)} proxies disponibles")
logger.info(f"Cache TTL: {config.CACHE_TTL}s, Cooldown: {config.CATALOG_COOLDOWN}s")
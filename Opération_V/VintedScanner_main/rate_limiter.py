import time
import logging
from threading import Lock

# Configuration du logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger('RateLimiter')


class RateLimiter:
    """Système avancé de limitation de taux avec backoff exponentiel"""

    def __init__(self, max_requests=5, period=1.0, max_retries=3, backoff_factor=2):
        """
        max_requests: Nombre maximum de requêtes par période
        period: Durée de la période en secondes
        max_retries: Nombre maximum de tentatives en cas d'échec
        backoff_factor: Facteur d'augmentation du délai entre les tentatives
        """
        self.max_requests = max_requests
        self.period = period
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.timestamps = []
        self.lock = Lock()

    def wait(self):
        """Attend si nécessaire pour respecter la limite de taux"""
        with self.lock:
            now
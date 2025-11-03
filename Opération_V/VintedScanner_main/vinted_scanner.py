import re
import time
import logging
import asyncio
import aiohttp
from datetime import datetime
from urllib.parse import urljoin
import requests
from json import JSONDecodeError
from collections import defaultdict
from .config import config
from .cache_manager import CacheManager
import concurrent.futures
import random
import json
from bs4 import BeautifulSoup
import os
import backoff

############################################
#  INITIALISATION DES COMPOSANTS SAAS      #
############################################

# Configuration avancée du logger
logger = logging.getLogger('vinted_scanner')
logger.setLevel(logging.INFO if not config.DEBUG else logging.DEBUG)

# Handler pour envoi vers service de logs centralisé
if config.LOG_SERVICE_ENDPOINT:
    from logging.handlers import HTTPHandler

    log_handler = HTTPHandler(
        config.LOG_SERVICE_ENDPOINT,
        method='POST',
        secure=True
    )
    logger.addHandler(log_handler)

# Initialisation des caches avec expiration
cache_manager_temp = CacheManager(filename="vinted_temp.json", ttl=config.CACHE_TTL)
cache_manager = CacheManager(filename=config.CATALOG_FILE, ttl=config.CACHE_TTL)

############################################
#  CONFIGURATION DES REQUÊTES HTTP         #
############################################

# Configuration des headers avec rotation User-Agent
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
]

# Pool de proxies rotatifs
PROXY_POOL = config.PROXY_POOL if hasattr(config, 'PROXY_POOL') else []


def get_random_headers():
    """Retourne des headers aléatoires avec des valeurs réalistes"""
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
        "DNT": "1",
        "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
        "Accept": "application/json",
        "Referer": "https://www.vinted.fr/",
        "Connection": "keep-alive"
    }


############################################
#  MODÈLES DE DONNÉES                      #
############################################

class CategoryNode:
    """Représente un nœud dans l'arborescence des catégories"""

    def __init__(self, id, name, parent_id=None):
        self.id = id
        self.name = name
        self.parent_id = parent_id
        self.children = []
        self.is_orphan = False  # Pour suivre les catégories orphelines

    def add_child(self, child_node):
        self.children.append(child_node)

    def to_dict(self):
        """Convertit le nœud en dictionnaire récursif"""
        return {
            "id": self.id,
            "name": self.name,
            "is_orphan": self.is_orphan,
            "children": [child.to_dict() for child in sorted(self.children, key=lambda x: x.name)]
        }


############################################
#  GESTION DES SESSIONS HTTP               #
############################################

class VintedSession:
    """Gestionnaire de session avec cookies dynamiques et rotation de proxies"""

    def __init__(self):
        self.cookies = {}
        self.last_refresh = 0
        self.refresh_interval = 600  # 10 minutes
        self.refresh_lock = False
        self.current_proxy = None
        self.proxy_index = 0
        self.refresh_cookies()

    def get_next_proxy(self):
        """Rotation des proxies avec fallback"""
        if not PROXY_POOL:
            return None

        if self.proxy_index >= len(PROXY_POOL):
            self.proxy_index = 0

        proxy = PROXY_POOL[self.proxy_index]
        self.proxy_index += 1
        return proxy

    def refresh_cookies(self):
        """Récupère de nouveaux cookies via Selenium avec gestion des proxies"""
        if self.refresh_lock:
            logger.warning("Refresh déjà en cours")
            return False

        try:
            self.refresh_lock = True
            logger.info("🔄 Mise à jour des cookies Vinted...")

            from vinted_cookie import fetch_vinted_cookies

            proxy = self.get_next_proxy()
            new_cookies = fetch_vinted_cookies(
                headless=True,
                proxy=proxy,
                user_agent=random.choice(USER_AGENTS)
            )

            if new_cookies:
                self.cookies = new_cookies
                self.last_refresh = time.time()
                logger.info(f"✅ {len(self.cookies)} cookies mis à jour via proxy: {proxy or 'Aucun'}")
                return True
            else:
                logger.error("❌ Échec de la récupération des cookies")
                return False
        except Exception as e:
            logger.error(f"🔥 Erreur rafraîchissement cookies: {str(e)}", exc_info=True)
            return False
        finally:
            self.refresh_lock = False

    def ensure_valid_cookies(self):
        """Vérifie et rafraîchit les cookies si nécessaire"""
        current_time = time.time()

        # Rafraîchissement périodique
        if current_time - self.last_refresh > self.refresh_interval:
            logger.info("🔄 Rafraîchissement périodique des cookies")
            return self.refresh_cookies()

        # Vérifie la présence des cookies essentiels
        required_cookies = {'_vinted_fr_session', 'datadome'}
        if not required_cookies.issubset(self.cookies.keys()):
            logger.warning("⚠️ Cookies essentiels manquants, rafraîchissement")
            return self.refresh_cookies()

        return True


# Session globale pour tout le module
vinted_session = VintedSession()


############################################
#  UTILITAIRES HTTP ASYNCHRONES           #
############################################

@backoff.on_exception(
    backoff.expo,
    (aiohttp.ClientError, asyncio.TimeoutError),
    max_tries=5,
    jitter=backoff.full_jitter(2)
)
async def async_fetch_json(session, url, params=None, headers=None, cooldown=0):
    """Version asynchrone pour récupérer du JSON avec gestion robuste des erreurs"""
    if cooldown > 0:
        await asyncio.sleep(cooldown)

    proxy = vinted_session.get_next_proxy()
    try:
        async with session.get(
                url,
                params=params,
                headers=headers,
                cookies=vinted_session.cookies,
                proxy=proxy,
                timeout=aiohttp.ClientTimeout(total=20)
        ) as response:

            if response.status == 200:
                return await response.json()
            elif response.status == 403:
                logger.warning("🔒 Accès refusé (403), mise à jour des cookies...")
                vinted_session.refresh_cookies()
                raise aiohttp.ClientResponseError("Require new cookies")
            elif response.status == 429:
                logger.warning("⏱️ Rate limit atteint - Backoff appliqué")
                raise aiohttp.ClientResponseError("Rate limited")
            else:
                logger.error(f"❌ Échec requête: {url} | Status: {response.status}")
                return None

    except Exception as e:
        logger.error(f"🔥 Erreur requête asynchrone: {str(e)}", exc_info=True)
        raise


def fetch_json(url, params=None, max_attempts=3, cooldown=1.0):
    """Wrapper synchrone pour compatibilité avec le code existant"""
    return asyncio.run(async_fetch_json_wrapper(url, params, max_attempts, cooldown))


async def async_fetch_json_wrapper(url, params, max_attempts, cooldown):
    """Wrapper asynchrone pour la fonction fetch_json"""
    headers = get_random_headers()
    async with aiohttp.ClientSession(headers=headers) as session:
        for attempt in range(1, max_attempts + 1):
            try:
                vinted_session.ensure_valid_cookies()
                result = await async_fetch_json(session, url, params, cooldown=cooldown)
                if result is not None:
                    return result
            except Exception as e:
                logger.warning(f"⚠️ Tentative {attempt}/{max_attempts} échouée: {str(e)}")
                await asyncio.sleep(cooldown * attempt)

        logger.error(f"❌ Échec après {max_attempts} tentatives sur {url}")
        return None


############################################
#  GESTION DU CATALOGUE                   #
############################################

def get_catalog_age():
    """
    Renvoie l'âge du catalogue en secondes
    (timestamp du cache → maintenant)

    Returns:
        int: Âge en secondes ou 9999999 en cas d'erreur
    """
    try:
        data = cache_manager.load()
        if not data or 'timestamp' not in data:
            logger.warning("⚠️ Catalogue non trouvé ou sans timestamp")
            return 9999999  # Valeur d'erreur standard
        return int(time.time() - data['timestamp'])
    except Exception as e:
        logger.error(f"🔥 Erreur get_catalog_age: {str(e)}", exc_info=True)
        return 9999999


def find_similar_items(item_url):
    """
    Trouve des articles similaires à partir d'une URL d'article Vinted
    Version optimisée pour production SaaS

    Args:
        item_url (str): URL complète de l'article Vinted

    Returns:
        list: Articles similaires ou liste vide en cas d'erreur
    """
    try:
        # Extraction de l'ID d'article depuis l'URL
        item_id = re.search(r'/items/(\d+)', item_url)
        if not item_id:
            logger.error(f"❌ URL d'article invalide: {item_url}")
            return []

        # Récupération des détails de l'article
        item_data = fetch_json(f"https://www.vinted.fr/api/v2/items/{item_id.group(1)}")
        if not item_data or 'item' not in item_data:
            return []

        item = item_data['item']

        # Construction des paramètres de recherche similaires
        search_params = {
            "brand_ids": [item['brand_id']] if item.get('brand_id') else None,
            "size_ids": [item['size_id']] if item.get('size_id') else None,
            "material_ids": [item['material_id']] if item.get('material_id') else None,
            "status_ids": [item['status_id']] if item.get('status_id') else None,
            "color_ids": [item['color_id']] if item.get('color_id') else None,
            "category_ids": [item['catalog_id']] if item.get('catalog_id') else None,
            "max_items": 10
        }

        return VintedSearch.search_items(**search_params)

    except Exception as e:
        logger.error(f"🔥 Erreur recherche similaire: {str(e)}", exc_info=True)
        return []


############################################
#  CONSTRUCTION DE L'ARBORESCENCE         #
############################################

def build_category_tree(categories):
    """Construit l'arborescence des catégories avec détection des orphelins"""
    # Créer un dictionnaire pour accéder rapidement aux catégories par ID
    categories_by_id = {}
    # Dictionnaire pour regrouper les enfants par parent_id
    children_by_parent = defaultdict(list)

    # Première passe: créer tous les nœuds
    for cat in categories:
        if not cat.get('id') or not cat.get('title'):
            continue  # Ignorer les entrées invalides

        node = CategoryNode(
            id=cat['id'],
            name=cat['title'],
            parent_id=cat.get('parent_id')
        )
        categories_by_id[cat['id']] = node

        parent_id = cat.get('parent_id')
        if parent_id:
            children_by_parent[parent_id].append(cat['id'])

    # Deuxième passe: construire l'arborescence
    root_categories = []
    orphaned_categories = []

    for cat_id, node in categories_by_id.items():
        parent_id = node.parent_id
        if parent_id is None:
            root_categories.append(node)
        else:
            parent_node = categories_by_id.get(parent_id)
            if parent_node is not None:
                parent_node.add_child(node)
            else:
                node.is_orphan = True
                orphaned_categories.append(cat_id)

    # Vérification de l'intégrité de l'arborescence
    total_nodes = 0
    stack = list(root_categories)
    while stack:
        current = stack.pop()
        total_nodes += 1
        stack.extend(current.children)

    logger.info(f"🔍 Vérification arborescence: {total_nodes} nœuds / {len(categories_by_id)} catégories")

    if orphaned_categories:
        logger.warning(f"⚠️ {len(orphaned_categories)} catégories orphelines: {orphaned_categories}")
    if total_nodes != len(categories_by_id):
        logger.warning(f"⚠️ Arborescence incomplète: {len(categories_by_id) - total_nodes} catégories manquantes")

    # Trier les catégories par nom
    root_categories.sort(key=lambda x: x.name)
    for node in categories_by_id.values():
        node.children.sort(key=lambda x: x.name)

    return root_categories


############################################
#  SCRAPING DES DONNÉES                   #
############################################

async def async_scrape_categories():
    """Récupère toutes les catégories et construit l'arborescence (asynchrone)"""
    BASE_URL = "https://www.vinted.fr"
    logger.info("🔄 Récupération de toutes les catégories...")

    url = f"{BASE_URL}/api/v2/catalogs"
    headers = get_random_headers()

    async with aiohttp.ClientSession(headers=headers) as session:
        data = await async_fetch_json(
            session,
            url,
            cooldown=0.5
        )

    if not data or 'catalogs' not in data:
        logger.error("❌ Échec de la récupération des catégories")
        return []

    categories = data['catalogs']

    # Nettoyage des données
    valid_categories = [
        cat for cat in categories
        if cat.get('id') and cat.get('title')
    ]

    logger.info(f"✅ {len(valid_categories)} catégories valides récupérées")

    # Construction de l'arborescence
    return build_category_tree(valid_categories)


def parse_filters_data(filters_data):
    """Parse les données de filtres Vinted avec gestion robuste des formats"""
    parsed_filters = {}

    try:
        if not filters_data:
            return parsed_filters

        # Cas 1: Structure principale dans un dictionnaire avec clé 'filters'
        if isinstance(filters_data, dict) and 'filters' in filters_data:
            filters = filters_data['filters']

            # Sous-cas 1: 'filters' est un dictionnaire
            if isinstance(filters, dict):
                for filter_id, filter_details in filters.items():
                    if not isinstance(filter_details, dict):
                        continue

                    options = filter_details.get('options', [])
                    if not isinstance(options, list):
                        continue

                    parsed_filters[filter_id] = {
                        "id": filter_id,
                        "title": filter_details.get("title", filter_id),
                        "options": options
                    }

            # Sous-cas 2: 'filters' est une liste
            elif isinstance(filters, list):
                for filter_entry in filters:
                    if not isinstance(filter_entry, dict):
                        continue

                    filter_id = filter_entry.get('id')
                    if not filter_id:
                        continue

                    options = filter_entry.get('options', [])
                    if not isinstance(options, list):
                        continue

                    parsed_filters[filter_id] = {
                        "id": filter_id,
                        "title": filter_entry.get("title", filter_id),
                        "options": options
                    }

        # Cas 2: Structure où les filtres sont directement dans une liste
        elif isinstance(filters_data, list):
            for filter_entry in filters_data:
                if not isinstance(filter_entry, dict):
                    continue

                filter_id = filter_entry.get('id')
                if not filter_id:
                    continue

                options = filter_entry.get('options', [])
                if not isinstance(options, list):
                    continue

                parsed_filters[filter_id] = {
                    "id": filter_id,
                    "title": filter_entry.get("title", filter_id),
                    "options": options
                }

        # Cas 3: Structure inconnue - tentative de récupération
        elif isinstance(filters_data, dict):
            for filter_id, filter_details in filters_data.items():
                if not isinstance(filter_details, dict):
                    continue

                options = filter_details.get('options', [])
                if not isinstance(options, list):
                    continue

                parsed_filters[filter_id] = {
                    "id": filter_id,
                    "title": filter_details.get("title", filter_id),
                    "options": options
                }

        logger.info(f"✅ {len(parsed_filters)} filtres parsés")
        return parsed_filters

    except Exception as e:
        logger.error(f"🔥 Erreur critique dans parse_filters_data: {str(e)}", exc_info=True)
        return {}


async def async_scrape_filters(session, category_id):
    """Récupère les filtres pour une catégorie spécifique (asynchrone)"""
    BASE_URL = "https://www.vinted.fr"
    logger.debug(f"🔍 Récupération des filtres pour la catégorie {category_id}")

    try:
        filters_data = await async_fetch_json(
            session,
            f"{BASE_URL}/api/v2/catalog/filters?catalog_ids[]={category_id}",
            cooldown=0.3
        )

        if not filters_data:
            logger.warning(f"⚠️ Aucune donnée de filtre pour la catégorie {category_id}")
            return {}

        return parse_filters_data(filters_data)

    except Exception as e:
        logger.error(f"🔥 Erreur récupération filtres catégorie {category_id}: {str(e)}", exc_info=True)
        return {}


async def async_scrape_all_filters(category_ids):
    """Récupère les filtres pour toutes les catégories en parallèle (asynchrone)"""
    logger.info(f"🔄 Récupération des filtres pour {len(category_ids)} catégories...")
    filters = {}

    async with aiohttp.ClientSession(headers=get_random_headers()) as session:
        tasks = [async_scrape_filters(session, cat_id) for cat_id in category_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            cat_id = category_ids[i]
            if isinstance(result, Exception):
                logger.error(f"❌ Erreur pour catégorie {cat_id}: {str(result)}")
            elif result:
                filters[cat_id] = result

    logger.info(f"✅ Filtres récupérés pour {len(filters)} catégories")
    return filters


def parse_brands_data(data):
    """Parse les données de marques avec fallback robuste"""
    try:
        brands = []

        # Cas 1: Données dans un dictionnaire avec clé 'brands'
        if isinstance(data, dict) and 'brands' in data:
            brands_list = data['brands']
            if isinstance(brands_list, list):
                for brand in brands_list:
                    if not isinstance(brand, dict):
                        continue
                    brand_id = brand.get('id')
                    brand_name = brand.get('name') or brand.get('title')
                    if brand_id and brand_name:
                        brands.append({
                            "id": brand_id,
                            "name": brand_name
                        })

        # Cas 2: Données directement dans une liste
        elif isinstance(data, list):
            for brand in data:
                if not isinstance(brand, dict):
                    continue
                brand_id = brand.get('id')
                brand_name = brand.get('name') or brand.get('title')
                if brand_id and brand_name:
                    brands.append({
                        "id": brand_id,
                        "name": brand_name
                    })

        return brands

    except Exception as e:
        logger.error(f"🔥 Erreur critique dans parse_brands_data: {str(e)}", exc_info=True)
        return []


async def async_scrape_brands():
    """Récupère toutes les marques avec pagination (asynchrone)"""
    BASE_URL = "https://www.vinted.fr"
    all_brands = []
    page = 1
    total_pages = 1

    async with aiohttp.ClientSession(headers=get_random_headers()) as session:
        while page <= total_pages:
            logger.info(f"📦 Récupération des marques - page {page}/{total_pages}")

            try:
                url = f"{BASE_URL}/api/v2/brands?page={page}&per_page=100"
                data = await async_fetch_json(session, url, cooldown=0.5)

                if not data:
                    logger.warning(f"⚠️ Aucune donnée pour la page {page} des marques")
                    break

                # Extraction des marques
                page_brands = parse_brands_data(data)
                all_brands.extend(page_brands)

                # Mise à jour de la pagination
                if isinstance(data, dict) and data.get('pagination'):
                    pagination = data['pagination']
                    total_pages = pagination.get('total_pages', 1)
                    current_page = pagination.get('current_page', page)
                    page = current_page + 1
                else:
                    page += 1

                # Vérification de la progression
                if page > total_pages:
                    break

                # Cooldown anti-rate-limit
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"❌ Erreur récupération marques page {page}: {str(e)}")
                break

    # Trier les marques par nom
    all_brands.sort(key=lambda x: x['name'])

    logger.info(f"✅ {len(all_brands)} marques valides récupérées")
    return all_brands


async def async_scrape_global_filters():
    """Récupère les filtres globaux avec fallback robuste"""
    BASE_URL = "https://www.vinted.fr"
    logger.info("🔄 Récupération des filtres globaux...")

    headers = get_random_headers()
    async with aiohttp.ClientSession(headers=headers) as session:
        try:
            filters_data = await async_fetch_json(
                session,
                f"{BASE_URL}/api/v2/catalog/filters",
                cooldown=0.5
            )

            if not filters_data:
                logger.warning("⚠️ Aucune donnée pour les filtres globaux")
                return {}

            return parse_filters_data(filters_data)

        except Exception as e:
            logger.error(f"🔥 Erreur critique dans scrape_global_filters: {str(e)}", exc_info=True)
            return {}


def clean_catalog_data(catalog_data):
    """Nettoie et optimise les données du catalogue avec gestion d'erreur granulaire"""
    try:
        # Vérification de la structure de base
        if not isinstance(catalog_data, dict):
            logger.error("🔥 Structure de données invalide pour le nettoyage")
            return {"timestamp": time.time()}

        cleaned_data = {
            "timestamp": catalog_data.get("timestamp", time.time()),
            "categories": [],
            "filters": {},
            "brands": [],
            "global_filters": {},
            "metadata": {
                "version": "2.0",
                "generated_at": datetime.utcnow().isoformat()
            }
        }

        # Nettoyage des catégories
        if "categories" in catalog_data and isinstance(catalog_data["categories"], list):
            cleaned_data["categories"] = catalog_data["categories"]

        # Nettoyage des marques
        if "brands" in catalog_data and isinstance(catalog_data["brands"], list):
            cleaned_data["brands"] = [
                brand for brand in catalog_data["brands"]
                if isinstance(brand, dict) and brand.get("id") and brand.get("name")
            ]
            cleaned_data["brands"].sort(key=lambda x: x["name"])

        # Nettoyage des filtres par catégorie
        if "filters" in catalog_data and isinstance(catalog_data["filters"], dict):
            for cat_id, filters in catalog_data["filters"].items():
                cleaned_filters = {}
                if not isinstance(filters, dict):
                    continue

                for filter_key, filter_data in filters.items():
                    try:
                        # Gestion des formats de filtres
                        if isinstance(filter_data, dict):
                            options = filter_data.get('options') or filter_data.get('values', [])
                        elif isinstance(filter_data, list):
                            options = filter_data
                        else:
                            logger.warning(f"⚠️ Format de filtre inattendu: {type(filter_data)}")
                            options = []

                        if isinstance(options, list):
                            # Nettoyage des options vides
                            cleaned_options = [
                                opt for opt in options
                                if isinstance(opt, dict) and opt.get("id") and opt.get("title")
                            ]
                            cleaned_options.sort(key=lambda x: x.get("title", ""))

                            if cleaned_options:
                                cleaned_filter = {
                                    "id": filter_key,
                                    "title": filter_data.get("title", filter_key) if isinstance(filter_data,
                                                                                                dict) else filter_key,
                                    "options": cleaned_options
                                }
                                cleaned_filters[filter_key] = cleaned_filter
                    except Exception as e:
                        logger.error(f"🔥 Erreur nettoyage filtre {filter_key} catégorie {cat_id}: {str(e)}",
                                     exc_info=True)

                if cleaned_filters:
                    cleaned_data["filters"][cat_id] = cleaned_filters

        # Nettoyage des filtres globaux
        if "global_filters" in catalog_data and isinstance(catalog_data["global_filters"], dict):
            cleaned_filters = {}
            for filter_key, filter_data in catalog_data["global_filters"].items():
                try:
                    if isinstance(filter_data, dict):
                        options = filter_data.get('options') or filter_data.get('values', [])
                    elif isinstance(filter_data, list):
                        options = filter_data
                    else:
                        logger.warning(f"⚠️ Format de filtre inattendu: {type(filter_data)}")
                        options = []

                    if isinstance(options, list):
                        cleaned_options = [
                            opt for opt in options
                            if isinstance(opt, dict) and opt.get("id") and opt.get("title")
                        ]
                        cleaned_options.sort(key=lambda x: x.get("title", ""))

                        if cleaned_options:
                            cleaned_filter = {
                                "id": filter_key,
                                "title": filter_data.get("title", filter_key) if isinstance(filter_data,
                                                                                            dict) else filter_key,
                                "options": cleaned_options
                            }
                            cleaned_filters[filter_key] = cleaned_filter
                except Exception as e:
                    logger.error(f"🔥 Erreur nettoyage filtre global {filter_key}: {str(e)}", exc_info=True)

            cleaned_data["global_filters"] = cleaned_filters

        logger.info(f"🧹 Catalogue nettoyé: {len(cleaned_data['brands'])} marques, "
                    f"{len(cleaned_data['filters'])} filtres catégoriels, "
                    f"{len(cleaned_data['global_filters'])} filtres globaux")

        return cleaned_data

    except Exception as e:
        logger.critical(f"🔥🔥 Erreur critique dans clean_catalog_data: {str(e)}", exc_info=True)
        return {"timestamp": time.time()}


async def async_scrape_live_catalog():
    """
    Récupère TOUTES les données brutes du catalogue Vinted (asynchrone)
    Retourne un dictionnaire structuré avec statut d'exécution
    """
    final_data = {
        "timestamp": time.time(),
        "categories": [],
        "filters": {},
        "brands": [],
        "global_filters": {},
        "status": "success",
        "errors": []
    }

    try:
        # 1. Récupération des catégories
        categories_tree = await async_scrape_categories()
        if categories_tree is None:
            logger.warning("⚠️ Aucune catégorie récupérée")
            categories_tree = []
        final_data["categories"] = [cat.to_dict() for cat in categories_tree]

        # Extraire tous les IDs de catégorie
        all_category_ids = set()
        stack = list(categories_tree)
        while stack:
            current = stack.pop()
            all_category_ids.add(current.id)
            stack.extend(current.children)

        logger.info(f"🌳 Arborescence construite: {len(all_category_ids)} catégories")

        # 2. Récupération des filtres par catégorie
        if all_category_ids:
            final_data["filters"] = await async_scrape_all_filters(list(all_category_ids))
        else:
            logger.warning("⚠️ Aucun ID de catégorie disponible - skip des filtres")

        # 3. Récupération des marques
        brands_list = await async_scrape_brands()
        if not brands_list:
            logger.warning("⚠️ Aucune marque récupérée")
        final_data["brands"] = brands_list

        # 4. Récupération des filtres globaux
        global_filters = await async_scrape_global_filters()
        if not global_filters:
            logger.warning("⚠️ Aucun filtre global récupéré")
        final_data["global_filters"] = global_filters

        # Nettoyage final
        final_data = clean_catalog_data(final_data)
        final_data["status"] = "success"
        logger.info("✅ Scraping complet réussi!")

    except Exception as e:
        logger.critical(f"🔥🔥 Échec scraping catalogue: {str(e)}", exc_info=True)
        final_data["status"] = "partial" if final_data.get("categories") else "failed"
        final_data["error"] = str(e)

    return final_data


async def async_update_catalog():
    """Met à jour le catalogue de manière asynchrone"""
    try:
        # Vérification du cooldown
        temp_data = cache_manager_temp.load()
        if temp_data is None:
            logger.info("ℹ️ Pas de fichier temporaire - pas de cooldown")
        elif 'timestamp' in temp_data:
            last_update = temp_data['timestamp']
            cooldown = getattr(config, 'CATALOG_COOLDOWN', 600)
            if time.time() - last_update < cooldown:
                logger.info(f"⏳ Cooldown actif ({cooldown}s) - annulation")
                return False
        else:
            logger.warning("⚠️ Fichier temporaire sans timestamp - pas de cooldown")

        # Scraping des données brutes
        logger.info("🚀 Lancement du scraping asynchrone du catalogue...")
        catalog_raw = await async_scrape_live_catalog()

        if not catalog_raw or catalog_raw.get("status") != "success":
            logger.error("❌ Échec du scraping du catalogue")
            return False

        # Sauvegarde des données brutes
        cache_manager_temp.save(catalog_raw)
        logger.info("💾 Catalogue brut sauvegardé")
        return True

    except Exception as e:
        logger.critical(f"🔥🔥 Erreur critique mise à jour: {str(e)}", exc_info=True)
        return False


def update_catalog():
    """Wrapper synchrone pour la mise à jour du catalogue"""
    return asyncio.run(async_update_catalog())


def load_catalog(filename="vinted_data.json", ttl=None):
    """
    Charge le catalogue depuis le fichier JSON (fichier par défut vinted_data.json)

    Returns:
        dict: Données du catalogue ou dict vide si erreur
    """
    cache = CacheManager(filename=filename, ttl=ttl or 3600)
    return cache.load() or {}


############################################
#  MOTEUR DE RECHERCHE                    #
############################################

class VintedSearch:
    """Classe professionnelle pour la recherche d'articles sur Vinted"""

    BASE_API_URL = "https://www.vinted.fr/api/v2/catalog/items"
    MAX_ITEMS_PER_REQUEST = 100
    MAX_CONCURRENT_REQUESTS = 5
    RATE_LIMIT_COOLDOWN = 1.0  # secondes entre les requêtes

    @staticmethod
    def build_search_params(
            search_text="",
            category_ids=None,
            brand_ids=None,
            size_ids=None,
            color_ids=None,
            material_ids=None,
            status_ids=None,
            price_from=0,
            price_to=1000,
            currency="EUR",
            order="newest_first",
            page=1,
            per_page=100
    ):
        """Construit les paramètres de recherche avec validation robuste"""
        params = {
            "search_text": search_text,
            "catalog_ids": category_ids or [],
            "brand_ids": brand_ids or [],
            "size_ids": size_ids or [],
            "color_ids": color_ids or [],
            "material_ids": material_ids or [],
            "status_ids": status_ids or [],
            "price_from": max(0, price_from),
            "price_to": min(10000, price_to) if price_to > 0 else 10000,
            "currency": currency,
            "order": order,
            "page": max(1, page),
            "per_page": min(per_page, VintedSearch.MAX_ITEMS_PER_REQUEST)
        }

        # Nettoyage des listes vides
        return {k: v for k, v in params.items() if v}

    @staticmethod
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError, aiohttp.ClientResponseError),
        max_tries=5,
        jitter=backoff.full_jitter(2),
        logger=logger
    )
    async def async_search_page(session, params):
        """Exécute une requête de recherche pour une page spécifique"""
        try:
            # Rotation des proxies et validation des cookies
            vinted_session.ensure_valid_cookies()
            proxy = vinted_session.get_next_proxy()

            async with session.get(
                    VintedSearch.BASE_API_URL,
                    params=params,
                    headers=get_random_headers(),
                    cookies=vinted_session.cookies,
                    proxy=proxy,
                    timeout=aiohttp.ClientTimeout(total=15)
            ) as response:

                # Gestion des codes d'erreur
                if response.status == 200:
                    return await response.json()
                elif response.status == 403:
                    logger.warning("🔒 Accès refusé (403), mise à jour des cookies...")
                    vinted_session.refresh_cookies()
                    raise aiohttp.ClientResponseError(
                        response.request_info,
                        response.history,
                        status=403,
                        message="Forbidden"
                    )
                elif response.status == 429:
                    logger.warning("⏱️ Rate limit atteint (429) - Backoff appliqué")
                    raise aiohttp.ClientResponseError(
                        response.request_info,
                        response.history,
                        status=429,
                        message="Too Many Requests"
                    )
                else:
                    logger.error(f"❌ Échec requête: Status {response.status}")
                    return None

        except Exception as e:
            logger.error(f"🔥 Erreur recherche page {params.get('page', 1)}: {str(e)}")
            raise

    @staticmethod
    async def async_search_items(
            search_params,
            max_items=100,
            max_workers=MAX_CONCURRENT_REQUESTS
    ):
        """Recherche asynchrone avec pagination et parallélisation"""
        all_items = []
        total_pages = 1
        current_page = 1
        params = VintedSearch.build_search_params(**search_params)

        async with aiohttp.ClientSession() as session:
            # Requête initiale pour obtenir la pagination
            first_page_params = params.copy()
            first_page_params["per_page"] = min(max_items, VintedSearch.MAX_ITEMS_PER_REQUEST)
            first_page_params["page"] = 1

            try:
                data = await VintedSearch.async_search_page(session, first_page_params)
                if not data:
                    return []

                items = data.get("items", [])
                all_items.extend(items)

                # Extraction des infos de pagination
                pagination = data.get("pagination", {})
                total_pages = pagination.get("total_pages", 1)
                current_page = pagination.get("current_page", 1)

                # Planification des requêtes parallèles pour les pages restantes
                if total_pages > 1 and len(all_items) < max_items:
                    # Calcul des pages nécessaires
                    pages_needed = min(total_pages - current_page,
                                       (max_items - len(all_items)) // VintedSearch.MAX_ITEMS_PER_REQUEST + 1)
                    page_range = range(current_page + 1, current_page + 1 + pages_needed)

                    # Création des tâches asynchrones
                    tasks = []
                    for page in page_range:
                        page_params = params.copy()
                        page_params["page"] = page
                        page_params["per_page"] = min(
                            VintedSearch.MAX_ITEMS_PER_REQUEST,
                            max_items - len(all_items)
                        )
                        tasks.append(VintedSearch.async_search_page(session, page_params))

                    # Exécution parallèle avec limitation
                    for future in asyncio.as_completed(tasks):
                        try:
                            data = await future
                            if data:
                                all_items.extend(data.get("items", []))
                                if len(all_items) >= max_items:
                                    # Annulation des tâches restantes
                                    for task in tasks:
                                        if not task.done():
                                            task.cancel()
                                    break
                        except Exception as e:
                            logger.warning(f"⚠️ Échec requête parallèle: {str(e)}")

                # Tronquer si on dépasse le max
                if len(all_items) > max_items:
                    all_items = all_items[:max_items]

            except Exception as e:
                logger.error(f"🔥 Erreur recherche principale: {str(e)}")

        return all_items

    @staticmethod
    def search_items(
            search_text="",
            category_ids=None,
            brand_ids=None,
            size_ids=None,
            color_ids=None,
            material_ids=None,
            status_ids=None,
            price_from=0,
            price_to=1000,
            currency="EUR",
            order="newest_first",
            max_items=100
    ):
        """
        Recherche synchrone d'articles avec pagination automatique

        Args:
            ... [paramètres détaillés] ...

        Returns:
            list: Articles correspondants ou liste vide si erreur
        """
        # Validation des paramètres
        max_items = max(1, min(1000, max_items))

        # Construction des paramètres
        params = {
            "search_text": search_text,
            "category_ids": category_ids,
            "brand_ids": brand_ids,
            "size_ids": size_ids,
            "color_ids": color_ids,
            "material_ids": material_ids,
            "status_ids": status_ids,
            "price_from": price_from,
            "price_to": price_to,
            "currency": currency,
            "order": order,
            "max_items": max_items
        }

        try:
            return asyncio.run(
                VintedSearch.async_search_items(params, max_items)
            )
        except Exception as e:
            logger.error(f"🔥 Erreur recherche synchrone: {str(e)}", exc_info=True)
            return []


############################################
#  API PUBLIQUE                           #
############################################

async def search_items(
        search_text="",
        category_ids=None,
        brand_ids=None,
        size_ids=None,
        color_ids=None,
        material_ids=None,
        status_ids=None,
        price_from=0,
        price_to=1000,
        currency="EUR",
        order="newest_first",
        max_items=100
):
    """
    Fonction globale pour la recherche asynchrone d'articles

    Args:
        ... [paramètres détaillés] ...

        Returns:
            list: Articles correspondants aux critères
    """
    search_params = {
        "search_text": search_text,
        "category_ids": category_ids,
        "brand_ids": brand_ids,
        "size_ids": size_ids,
        "color_ids": color_ids,
        "material_ids": material_ids,
        "status_ids": status_ids,
        "price_from": price_from,
        "price_to": price_to,
        "currency": currency,
        "order": order
    }
    return await VintedSearch.async_search_items(search_params, max_items)
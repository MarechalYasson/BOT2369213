import json
import time
import requests
import base64
import os
import logging
import re
import hashlib
import random
from dotenv import load_dotenv
from requests.exceptions import RequestException, Timeout, HTTPError

# Chargement des variables d'environnement
load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")  # Optimisation catalogue
DEEPSEEK_MODEL_TEXT = os.getenv("DEEPSEEK_MODEL_TEXT", "deepseek-chat")  # Génération de descriptions texte
DEEPSEEK_MODEL_VISION = os.getenv("DEEPSEEK_MODEL_VISION", "deepseek-vl")  # Vision → TODO

# Nouveaux modèles OpenAI
OPENAI_MODEL_CATALOG = os.getenv("OPENAI_MODEL_CATALOG", "gpt-4-turbo")
OPENAI_MODEL_TEXT = os.getenv("OPENAI_MODEL_TEXT", "gpt-3.5-turbo")
OPENAI_MODEL_VISION = os.getenv("OPENAI_MODEL_VISION", "gpt-4-vision-preview")

# Configuration du logger unique
logger = logging.getLogger('deepseek')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# API endpoints
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

TEMP_FILE = "vinted_temp.json"
FINAL_FILE = "vinted_data.json"
CACHE_DIR = ".cache"
CACHE_FILE = os.path.join(CACHE_DIR, "catalog_cache.json")

MAX_RETRIES = 3
MAX_PROMPT_LENGTH = 10000

# Prompt pour optimisation catalogue
PROMPT_OPTIMISATION_CATALOGUE = """
Tu es un expert du site Vinted et de l’optimisation de catalogues e-commerce.

Voici un catalogue brut contenant :
– Des catégories avec leurs IDs, noms et sous-catégories (« children »)
– Des marques avec ID et nom
– Des filtres globaux et par catégorie

⚙️ Ta mission est de nettoyer et optimiser ce catalogue sans perdre d’informations essentielles. Respecte STRICTEMENT les consignes suivantes :

───────────────────────────────
🔍 1. CATEGORIES
───────────────────────────────
- Conserve les champs « id » (entier) et « name » (chaîne de caractères).
- Supprime uniquement les catégories sans ID ET sans nom.
- Trie les catégories et leurs enfants par ordre alphabétique sur le champ « name ».
- Conserve la hiérarchie parent > enfant dans le champ « children ».
- Corrige les noms (supprime les espaces doublons, normalise la casse, enlève les caractères spéciaux inutiles).
- Ne modifie JAMAIS un ID existant.

───────────────────────────────
🔍 2. MARQUES (brands)
───────────────────────────────
- Chaque marque est un objet { "id": <entier ou null>, "name": <chaîne> }.
- Conserve toutes les marques, même si l’ID est absent (mettre "id": null).
- Trie les marques par ordre alphabétique du champ « name ».
- Supprime uniquement les doublons exacts (même « id » ET même « name »).
- Ne modifie PAS les IDs.

───────────────────────────────
🔍 3. FILTRES PAR CATÉGORIE (« filters ») et FILTRES GLOBAUX (« global_filters »)
───────────────────────────────
- Structure attendue : un objet JSON avec pour chaque catégorie un sous-objet par type de filtre.
- Types de filtres à traiter (même structure globale et locale) :
    - size, color, status, material, season, gender,
    - type (ex : T-shirts, Jeans),
    - style (ex : Streetwear, Classique),
    - pattern (ex : Uni, Rayé),
    - brand_type (ex : Luxe, Premium),
    - occasion (ex : Mariage, Sport),
    - tech_features (ex : Imperméable),
    - length (ex : Court, Long)

- Chaque filtre est une liste d’objets { "id": <int ou null>, "name": <string> }.
- Trie les options de chaque filtre par ordre alphabétique sur « name ».
- Ne supprime PAS les options qui possèdent un ID, même si le nom est vide.
- Supprime uniquement les filtres sans ID et sans nom.

───────────────────────────────
🔍 4. STRUCTURE GLOBALE ATTENDUE
───────────────────────────────
Réponds UNIQUEMENT avec le JSON final optimisé, dans cette structure :

{
  "timestamp": <timestamp_unix>,
  "categories": [ ... ],
  "brands": [ ... ],
  "filters": {
    "<category_id>": {
      "size": [ ... ],
      "color": [ ... ],
      "status": [ ... ],
      "material": [ ... ],
      "season": [ ... ],
      "gender": [ ... ],
      "type": [ ... ],
      "style": [ ... ],
      "pattern": [ ... ],
      "brand_type": [ ... ],
      "occasion": [ ... ],
      "tech_features": [ ... ],
      "length": [ ... ]
    }
  },
  "global_filters": {
      "size": [ ... ],
      "color": [ ... ],
      "status": [ ... ],
      "material": [ ... ],
      "season": [ ... ],
      "gender": [ ... ],
      "type": [ ... ],
      "style": [ ... ],
      "pattern": [ ... ],
      "brand_type": [ ... ],
      "occasion": [ ... ],
      "tech_features": [ ... ],
      "length": [ ... ]
  }
}

───────────────────────────────
⚠️ Règles finales impératives
───────────────────────────────
– Ne change PAS la structure des clés JSON.
– Ne réponds PAS avec du texte explicatif, seulement le JSON final.
– Tous les tableaux doivent être triés alphabétiquement par « name ».
– Ne fusionne PAS les filtres globaux et par catégorie.
– Ne change PAS les IDs.
"""

# ===== ARCHITECTURE UNIFIÉE AMÉLIORÉE =====
def call_ai_api(payload, endpoint="chat", fallback_model="gpt-4-turbo", response_format="text",
                model_env_key="DEEPSEEK_MODEL"):
    """Appel unifié aux APIs d'IA avec fallback intelligent"""
    # Détermination du timeout basé sur le modèle
    model_timeout = 180 if "gpt-4" in fallback_model else 90

    # Tentative DeepSeek en premier
    try:
        deepseek_payload = payload.copy()

        # Configuration spécifique pour Vision
        if endpoint == "vision":
            logger.info("🔄 Vision demandée : DeepSeek non disponible, utilisation de OpenAI Vision")
            return call_openai_vision_api(payload, fallback_model, timeout=model_timeout)

        # Sélection du modèle DeepSeek approprié
        model_map = {
            "DEEPSEEK_MODEL": DEEPSEEK_MODEL,
            "DEEPSEEK_MODEL_TEXT": DEEPSEEK_MODEL_TEXT,
            "DEEPSEEK_MODEL_VISION": DEEPSEEK_MODEL_VISION
        }
        deepseek_model = model_map.get(model_env_key, DEEPSEEK_MODEL)

        if "model" not in deepseek_payload:
            deepseek_payload["model"] = deepseek_model

        # Format JSON si nécessaire
        if response_format == "json_object":
            deepseek_payload["response_format"] = {"type": "json_object"}

        logger.info(f"Tentative DeepSeek ({deepseek_model})...")
        start_time = time.time()
        response = requests.post(
            DEEPSEEK_URL,
            json=deepseek_payload,
            headers=HEADERS,
            timeout=model_timeout
        )
        response.raise_for_status()

        result = response.json()
        if "choices" not in result or not result["choices"]:
            raise ValueError("Réponse DeepSeek invalide: clé 'choices' manquante")

        content = result["choices"][0]["message"]["content"].strip()
        duration = time.time() - start_time
        logger.info(f"✅ DeepSeek réussi en {duration:.2f}s")
        return content

    except Exception as e:
        logger.warning(f"DeepSeek échoué → fallback {fallback_model}: {str(e)}")

        # Fallback OpenAI
        try:
            if not OPENAI_API_KEY:
                raise RuntimeError("Clé OpenAI manquante")

            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }

            # Sélection automatique du modèle si non spécifié
            if fallback_model == "auto":
                fallback_model = OPENAI_MODEL_CATALOG if response_format == "json_object" else OPENAI_MODEL_TEXT

            openai_payload = {
                "model": fallback_model,
                "messages": payload["messages"],
                "temperature": payload.get("temperature", 0.7),
                "max_tokens": payload.get("max_tokens", 300)
            }

            # Correction cruciale: format JSON pour OpenAI
            if response_format == "json_object":
                # Format requis par OpenAI (string)
                openai_payload["response_format"] = "json_object"

            # Gestion spécifique pour Vision
            if endpoint == "vision":
                return call_openai_vision_api(payload, fallback_model, timeout=model_timeout)

            logger.info(f"Tentative OpenAI ({fallback_model})...")
            start_time = time.time()
            response = requests.post(
                OPENAI_URL,
                json=openai_payload,
                headers=headers,
                timeout=model_timeout
            )
            response.raise_for_status()

            result = response.json()
            if "choices" not in result or not result["choices"]:
                raise ValueError("Réponse OpenAI invalide: pas de choix")

            content = result["choices"][0]["message"]["content"].strip()
            duration = time.time() - start_time
            logger.info(f"✅ OpenAI réussi en {duration:.2f}s")
            return content

        except Exception as fallback_error:
            logger.error(f"Échec complet API: {str(fallback_error)}")
            raise RuntimeError(f"Échec DeepSeek + OpenAI: {str(fallback_error)}")


def call_openai_vision_api(payload, fallback_model, timeout=90):
    """Appelle OpenAI Vision (GPT-4 Vision) pour traiter une image en base64."""
    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }

        # Payload complet avec fallback_model (gpt-4-vision-preview normalement)
        openai_payload = {
            "model": fallback_model,
            "messages": payload.get("messages", []),
            "max_tokens": payload.get("max_tokens", 250),
            "temperature": payload.get("temperature", 0.7)
        }

        url = "https://api.openai.com/v1/chat/completions"
        logger.info(f"🔄 Tentative OpenAI Vision ({fallback_model})...")

        response = requests.post(url, headers=headers, json=openai_payload, timeout=timeout)
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]

    except Exception as e:
        logger.error(f"🔥 Erreur OpenAI Vision: {str(e)}", exc_info=True)
        raise



# ===== FONCTIONS EXISTANTES MODIFIÉES =====
def optimize_catalog_via_api(prompt):
    """Optimisation du catalogue via API avec fallback contrôlé"""
    try:
        payload = {
            "messages": [
                {"role": "system", "content": "Tu es un assistant expert Vinted."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 4096
        }

        optimized_json = call_ai_api(
            payload,
            endpoint="chat",
            fallback_model=OPENAI_MODEL_CATALOG,
            response_format="json_object",
            model_env_key="DEEPSEEK_MODEL"  # Utilise le modèle catalogue
        )

        # Extraction et validation
        parsed_data = extract_json_from_text(optimized_json)
        validate_json_structure(parsed_data)
        return parsed_data

    except Exception as e:
        logger.error(f"Échec traitement catalogue: {str(e)}")
        raise


def generer_description_depuis_image(image_path):
    """Génération de description à partir d'une image avec fallback"""
    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        payload = {
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_COMMUN},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{encoded_image}"}
                ]
            }],
            "temperature": 0.7,
            "max_tokens": 250
        }

        # Utilisation du fallback OpenAI Vision
        return call_ai_api(
            payload,
            endpoint="vision",
            fallback_model=OPENAI_MODEL_VISION
        )

    except Exception as e:
        logger.error(f"Erreur génération image: {str(e)}")
        return f"❌ Erreur lors de la génération de la description"


def generer_description_depuis_texte(nom_produit):
    """Génération de description à partir d'un texte avec fallback"""
    try:
        prompt = f"{PROMPT_COMMUN}\n\nProduit à décrire : {nom_produit}"

        payload = {
            "messages": [
                {"role": "system", "content": "Tu es un assistant expert Vinted."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 250
        }

        return call_ai_api(
            payload,
            endpoint="chat",
            fallback_model=OPENAI_MODEL_TEXT,
            model_env_key="DEEPSEEK_MODEL_TEXT"  # Utilise le modèle texte
        )

    except Exception as e:
        logger.error(f"Erreur génération texte: {str(e)}")
        return f"❌ Erreur lors de la génération de la description"


# ===== FONCTIONS EXISTANTES CORRIGÉES =====
def process_catalog_chunks(raw_data):
    """Traite le catalogue en chunks si nécessaire avec découpage intelligent"""
    full_prompt = build_catalog_prompt(raw_data)

    if len(full_prompt) <= MAX_PROMPT_LENGTH:
        logger.info("Prompt complet dans les limites. Traitement en une seule requête.")
        return optimize_catalog_via_api(full_prompt)

    logger.warning(f"Prompt trop long ({len(full_prompt)} > {MAX_PROMPT_LENGTH}). Découpage intelligent...")

    categories = raw_data.get('categories', [])
    brands = raw_data.get('brands', [])
    global_filters = raw_data.get('global_filters', {})

    merged_data = {
        "timestamp": int(time.time()),
        "categories": [],
        "brands": [],
        "filters": {},
        "global_filters": global_filters
    }

    current_chunk = []
    current_size = 0
    chunks = []

    for category in categories:
        cat_size = len(category.get('name', '')) + sum(
            len(child.get('name', '')) for child in category.get('children', [])[:5])

        if current_size + cat_size > MAX_PROMPT_LENGTH * 0.8 and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_size = 0

        current_chunk.append(category)
        current_size += cat_size

    if current_chunk:
        chunks.append(current_chunk)

    logger.info(f"Découpage en {len(chunks)} chunks basé sur la taille estimée")

    for i, chunk in enumerate(chunks):
        logger.info(f"Traitement du chunk {i + 1}/{len(chunks)} ({len(chunk)} catégories)")
        try:
            chunk_data = {
                "categories": chunk,
                "brands": brands,
                "global_filters": global_filters
            }
            chunk_prompt = build_catalog_prompt(chunk_data)
            optimized_chunk = optimize_catalog_via_api(chunk_prompt)

            merged_data['categories'].extend(optimized_chunk.get('categories', []))

            if i == 0:
                merged_data['brands'] = optimized_chunk.get('brands', [])
                merged_data['filters'] = optimized_chunk.get('filters', {})

        except Exception as e:
            logger.error(f"Échec du chunk {i + 1}: {str(e)}")
            logger.info("Poursuite du traitement avec les chunks restants")

    seen_categories = set()
    deduped_categories = []
    for cat in merged_data['categories']:
        cat_id = cat.get('id')
        cat_name = cat.get('name', '')
        identifier = f"{cat_id}-{cat_name}" if cat_id else cat_name

        if identifier and identifier not in seen_categories:
            seen_categories.add(identifier)
            deduped_categories.append(cat)
    merged_data['categories'] = deduped_categories

    # CORRECTION CRITIQUE: Déduplication correcte des marques
    brands_dict = {}
    for brand in merged_data['brands']:
        name = brand.get('name', '').lower()
        if name and name not in brands_dict:
            brands_dict[name] = brand

    merged_data['brands'] = sorted(
        brands_dict.values(),
        key=lambda x: x.get('name', '').lower()
    )

    logger.info(f"Fusion réussie: {len(merged_data['categories'])} catégories, {len(merged_data['brands'])} marques")
    return merged_data


# ===== FONCTIONS EXISTANTES CONSERVÉES =====
def get_retry_delay(error, attempt):
    """Retourne le délai de réessai intelligent selon le type d'erreur"""
    jitter = random.uniform(-5, 5)

    if isinstance(error, HTTPError):
        status_code = error.response.status_code
        if status_code == 429:
            return 90 + jitter
        elif status_code == 400:
            return None
        elif 500 <= status_code < 600:
            return 30 + jitter
    elif isinstance(error, Timeout):
        return 60 + jitter

    return (2 ** attempt) * 10 + jitter


def calculate_timeout(prompt_length, model):
    """Calcule dynamiquement le timeout"""
    base_timeout = 60
    size_factor = max(1, prompt_length / 1000)
    model_factor = 1.5 if "reasoner" in model else 1.0
    return min(300, int(base_timeout * size_factor * model_factor))


def validate_json_structure(data):
    """Valide la structure ET les types de données du JSON optimisé"""
    required_structure = {
        "timestamp": int,
        "categories": list,
        "brands": list,
        "filters": dict,
        "global_filters": dict
    }

    errors = []

    # Vérification des clés présentes
    missing_keys = [key for key in required_structure if key not in data]
    if missing_keys:
        errors.append(f"Clés manquantes: {', '.join(missing_keys)}")

    # Vérification des types de données
    for key, expected_type in required_structure.items():
        if key in data and not isinstance(data[key], expected_type):
            actual_type = type(data[key]).__name__
            errors.append(f"Type incorrect pour '{key}': attendu {expected_type.__name__}, obtenu {actual_type}")

    # Vérification spécifique des catégories
    if "categories" in data:
        for i, category in enumerate(data["categories"]):
            if not isinstance(category, dict):
                errors.append(f"Catégorie #{i + 1} n'est pas un objet")
            elif "name" not in category or not category["name"]:
                errors.append(f"Catégorie #{i + 1} manque un nom valide")

    if errors:
        raise ValueError("\n".join(errors))

    return True


def extract_json_from_text(text):
    """Tente d'extraire un JSON valide de différents formats"""
    # Essai 1: JSON brut
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Essai 2: JSON encapsulé dans markdown
    md_match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
    if md_match:
        try:
            return json.loads(md_match.group(1))
        except json.JSONDecodeError:
            pass

    # Essai 3: JSON encapsulé dans code
    code_match = re.search(r'```(.*?)```', text, re.DOTALL)
    if code_match:
        try:
            return json.loads(code_match.group(1))
        except json.JSONDecodeError:
            pass

    # Essai 4: Tout objet JSON dans le texte
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    # Essai 5: Format YAML (conversion simple)
    if ":" in text and "- " in text:
        try:
            fixed_text = text.replace("'", '"')
            fixed_text = re.sub(r'(\w+):', r'"\1":', fixed_text)
            fixed_text = re.sub(r':\s+([^"\s]+)', r': "\1"', fixed_text)
            return json.loads(f"{{{fixed_text}}}")
        except Exception:
            pass

    raise ValueError("Aucun format JSON valide détecté dans la réponse")


def optimize_catalog():
    """Orchestrateur principal pour l'optimisation du catalogue"""
    logger.info("Démarrage de l'optimisation du catalogue")
    logger.info(f"Modèle sélectionné: {DEEPSEEK_MODEL}")

    try:
        with open(TEMP_FILE, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            logger.info(f"Données brutes chargées : {len(raw_data.get('categories', []))} catégories")

        os.makedirs(CACHE_DIR, exist_ok=True)

        json_str = json.dumps(raw_data, sort_keys=True, ensure_ascii=False)
        current_hash = hashlib.sha256(json_str.encode('utf-8')).hexdigest()

        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)

                if cache_data.get('hash') == current_hash:
                    optimized_data = cache_data.get('data')
                    if optimized_data:
                        logger.info("Cache valide trouvé. Utilisation des données optimisées en cache.")
                        return save_final_data(optimized_data)
            except Exception as e:
                logger.warning(f"Erreur lors de la lecture du cache : {str(e)}")

    except FileNotFoundError:
        logger.error(f"Fichier {TEMP_FILE} introuvable")
        return False
    except json.JSONDecodeError:
        logger.error("Erreur de décodage JSON - fichier corrompu")
        return False
    except Exception as e:
        logger.error(f"Erreur lors du chargement du fichier : {str(e)}")
        return False

    optimized_data = None
    for attempt in range(MAX_RETRIES):
        try:
            optimized_data = process_catalog_chunks(raw_data)
            if optimized_data:
                break
        except Exception as e:
            logger.warning(f"Tentative {attempt + 1} échouée: {str(e)}")
            delay = get_retry_delay(e, attempt)
            if delay is None:
                logger.error("Arrêt des tentatives (erreur client irrécupérable)")
                break

            if attempt < MAX_RETRIES - 1:
                logger.info(f"Réessai dans {delay:.1f} secondes...")
                time.sleep(delay)

    if not optimized_data:
        logger.error("Échec de l'optimisation après %d tentatives", MAX_RETRIES)
        return False

    try:
        cache_data = {
            "hash": current_hash,
            "data": optimized_data,
            "timestamp": time.time()
        }
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False)
        logger.info("Cache mis à jour avec succès")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du cache : {str(e)}")

    return save_final_data(optimized_data)


def save_final_data(optimized_data):
    """Sauvegarde et validation des données optimisées"""
    try:
        validate_json_structure(optimized_data)

        with open(FINAL_FILE, 'w', encoding='utf-8') as f:
            json.dump(optimized_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Catalogue optimisé sauvegardé dans {FINAL_FILE}")
        return True
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde : {str(e)}")
        return False


def build_catalog_prompt(raw_data):
    """Construit un prompt texte clair à partir des données brutes du catalogue"""
    return PROMPT_OPTIMISATION_CATALOGUE


# Prompt pour descriptions produits
PROMPT_COMMUN = (
    "Tu es un expert en mode et ventes de vêtements sur Vinted.\n"
    "Rédige une description courte, naturelle, fluide et professionnelle.\n"
    "Indique : marque, couleur, coupe, style général (casual, sport, vintage, streetwear).\n"
    "Suggère une idée simple pour le porter (ex : jean, baskets).\n"
    "Précise : très bon état, prêt à porter.\n"
    "Si tu connais la taille, indique-la sous la forme 'Taille : M'. Sinon, n'indique rien.\n"
    "Termine toujours par deux phrases distinctes sur deux lignes :\n"
    "'Envoyé sous 24h, lavé et repassé.'\n"
    "'N'hésite pas si tu as des questions.'\n"
    "Pas d'emojis, pas de hashtags, pas de mise en forme spéciale.\n"
    "Fais des retours à la ligne pour une meilleure lisibilité."
)

if __name__ == "__main__":
    optimize_catalog()
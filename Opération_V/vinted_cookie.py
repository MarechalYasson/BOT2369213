import os
import logging
import time
from typing import Dict, Optional, Set, List, Tuple
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (TimeoutException,
                                        WebDriverException,
                                        NoSuchElementException,
                                        ElementNotInteractableException,
                                        InvalidSelectorException)
import undetected_chromedriver as uc
from dotenv import load_dotenv

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('vinted_cookie_fetcher')
logger.setLevel(logging.INFO)

# Charger les variables d'environnement
load_dotenv()


def is_logged_in(driver) -> bool:
    """Vérifie la présence d'éléments indiquant une session active avec des sélecteurs résilients"""
    indicators = [
        (By.CSS_SELECTOR, "[data-testid*='member-menu']"),  # Menu utilisateur
        (By.CSS_SELECTOR, "[href*='/member/']"),  # Lien profil
        (By.CSS_SELECTOR, "a[href*='/transactions']"),  # Lien transactions
        (By.CSS_SELECTOR, "a[href*='/favorites']"),  # Lien favoris
        (By.CSS_SELECTOR, "[aria-label*='Profil']"),  # Icone profil
        (By.CSS_SELECTOR, ".user-menu")  # Classe générique
    ]
    for by, selector in indicators:
        try:
            driver.find_element(by, selector)
            logger.debug(f"Élément de session détecté: {selector}")
            return True
        except NoSuchElementException:
            continue
    return False


def accept_cookie_banner(driver) -> bool:
    """Tente d'accepter la bannière cookies avec plusieurs stratégies"""
    selectors = [
        (By.CSS_SELECTOR, "#onetrust-accept-btn-handler"),  # Sélecteur principal
        (By.CSS_SELECTOR, "button#onetrust-accept-btn-handler"),  # Variante
        (By.CSS_SELECTOR, "button.cookie-consent__accept-button"),  # Autre variante
        (By.CSS_SELECTOR, "button[data-testid*='cookie-banner-accept']"),  # Nouveau format
        (By.CSS_SELECTOR, "button[class*='cookie-consent']"),  # Sélecteur générique
        (By.CSS_SELECTOR, "button[id*='accept-cookie']")  # Pattern ID
    ]

    for by, selector in selectors:
        try:
            accept_btn = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((by, selector)))
            accept_btn.click()
            logger.info("Bannière cookies acceptée")
            return True
        except (TimeoutException, ElementNotInteractableException):
            continue

    logger.debug("Pas de bannière cookies détectée")
    return False


def detect_blocking_page(driver) -> bool:
    """Détecte les pages de blocage ou CAPTCHA avec des sélecteurs valides"""
    blocking_indicators = [
        (By.CSS_SELECTOR, "div#captcha"),
        (By.CSS_SELECTOR, "div.challenge"),
        (By.CSS_SELECTOR, "div.security-page"),
        (By.CSS_SELECTOR, "iframe[src*='captcha']"),
        (By.XPATH, "//h1[contains(text(), 'Vérification')]")
    ]

    for by, selector in blocking_indicators:
        try:
            driver.find_element(by, selector)
            logger.warning(f"Page de blocage détectée : {selector}")
            return True
        except (NoSuchElementException, InvalidSelectorException):
            continue
    return False


def perform_login(driver) -> bool:
    """Effectue le processus de connexion avec résilience et détection d'erreurs"""
    logger.info("Tentative de connexion...")
    try:
        # Trouver le bouton de connexion avec plusieurs sélecteurs
        login_selectors = [
            (By.CSS_SELECTOR, "[data-testid*='login-button']"),
            (By.CSS_SELECTOR, "a[href*='/auth/login']"),
            (By.XPATH, "//button[contains(text(), 'Connexion')]"),
            (By.XPATH, "//a[contains(text(), 'Se connecter')]"),
            (By.CSS_SELECTOR, ".login-link")
        ]

        login_found = False
        for by, selector in login_selectors:
            try:
                login_btn = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((by, selector)))
                login_btn.click()
                login_found = True
                break
            except (TimeoutException, ElementNotInteractableException):
                continue

        if not login_found:
            raise TimeoutException("Aucun bouton de connexion trouvé")

        # Attendre le formulaire avec timeout dynamique
        form_selectors = [
            (By.CSS_SELECTOR, "form[data-testid*='login-form']"),
            (By.CSS_SELECTOR, "form#user_session"),
            (By.CSS_SELECTOR, "div.login-form")
        ]

        form_found = False
        for by, selector in form_selectors:
            try:
                WebDriverWait(driver, 10).until(
                    EC.visibility_of_element_located((by, selector)))
                form_found = True
                break
            except TimeoutException:
                continue

        if not form_found:
            raise TimeoutException("Formulaire de connexion non trouvé")

        # Remplir les identifiants avec sélecteurs flexibles
        username_selectors = [
            (By.CSS_SELECTOR, "input#user[data-testid]"),
            (By.CSS_SELECTOR, "input#user_session_login"),
            (By.CSS_SELECTOR, "input[name='user_session[login]']"),
            (By.CSS_SELECTOR, "input[type='email']")
        ]
        password_selectors = [
            (By.CSS_SELECTOR, "input#password[data-testid]"),
            (By.CSS_SELECTOR, "input#user_session_password"),
            (By.CSS_SELECTOR, "input[name='user_session[password]']"),
            (By.CSS_SELECTOR, "input[type='password']")
        ]

        # Trouver le champ username
        username_field = None
        for by, selector in username_selectors:
            try:
                username_field = driver.find_element(by, selector)
                break
            except NoSuchElementException:
                continue
        if not username_field:
            raise NoSuchElementException("Champ utilisateur non trouvé")

        # Trouver le champ password
        password_field = None
        for by, selector in password_selectors:
            try:
                password_field = driver.find_element(by, selector)
                break
            except NoSuchElementException:
                continue
        if not password_field:
            raise NoSuchElementException("Champ mot de passe non trouvé")

        username_field.send_keys(os.getenv("VINTED_USER"))
        password_field.send_keys(os.getenv("VINTED_PASS"))

        # Soumettre le formulaire
        submit_selectors = [
            (By.CSS_SELECTOR, "[data-testid*='login-form__submit-button']"),
            (By.CSS_SELECTOR, "button[type='submit']"),
            (By.CSS_SELECTOR, "input[name='commit']")
        ]

        submit_btn = None
        for by, selector in submit_selectors:
            try:
                submit_btn = driver.find_element(by, selector)
                break
            except NoSuchElementException:
                continue
        if not submit_btn:
            raise NoSuchElementException("Bouton de soumission non trouvé")

        submit_btn.click()

        # Vérifier la connexion avec plusieurs indicateurs
        WebDriverWait(driver, 15).until(
            lambda d: is_logged_in(d) or detect_blocking_page(d)
        )

        if not is_logged_in(driver):
            raise RuntimeError("Échec de connexion après soumission")

        logger.info("Connexion réussie")
        return True

    except Exception as e:
        logger.error(f"Échec de la connexion: {type(e).__name__} - {str(e)}")
        return False


def wait_for_cookies(driver, required_cookies: Set[str], timeout: int = 30) -> bool:
    """Attend que les cookies requis soient présents avec vérification anti-bot"""
    start_time = time.time()
    last_status = time.time()

    while time.time() - start_time < timeout:
        current_cookies = {c['name'] for c in driver.get_cookies()}

        # Vérifier si les cookies requis sont présents
        if required_cookies.issubset(current_cookies):
            return True

        # Vérifier périodiquement si un blocage est détecté
        if time.time() - last_status > 5:
            if detect_blocking_page(driver):
                logger.warning("Détection d'anti-bot - tentative échouée")
                return False
            last_status = time.time()

        time.sleep(1)

    # Log des cookies manquants
    missing = required_cookies - set(c['name'] for c in driver.get_cookies())
    logger.warning(f"Cookies manquants après attente: {', '.join(missing)}")
    return False


def fetch_vinted_cookies(
        headless: bool = True,
        max_retries: int = 3,
        proxy: Optional[str] = None,
        user_agent: Optional[str] = None
) -> Optional[Dict[str, str]]:
    """
    Récupère les cookies de session Vinted via un navigateur headless

    Args:
        headless: Mode sans affichage du navigateur
        max_retries: Nombre maximum de tentatives
        proxy: Proxy à utiliser (format: http://user:pass@host:port)
        user_agent: User-agent personnalisé à utiliser

    Returns:
        Dictionnaire des cookies ou None en cas d'échec
    """
    required_cookies = {'_vinted_fr_session', 'datadome'}
    driver = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Tentative {attempt}/{max_retries}")

            # Configuration avancée du navigateur
            options = uc.ChromeOptions()
            if headless:
                options.add_argument("--headless=new")
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1280,720")
            options.add_argument("--lang=fr-FR")
            options.add_argument("--disable-web-security")
            options.add_argument("--allow-running-insecure-content")

            # Configuration proxy si fourni
            if proxy:
                logger.info(f"Utilisation du proxy: {proxy.split('@')[-1] if '@' in proxy else proxy}")
                options.add_argument(f"--proxy-server={proxy}")

            # Paramètres pour éviter la détection
            options.add_argument("--disable-blink-features")
            options.add_argument("--disable-blink-features=AutomationControlled")


            # Initialisation du driver
            driver = uc.Chrome(
                options=options,
                use_subprocess=True,
                headless=headless
            )

            # Configuration des headers
            ua = user_agent or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
            driver.execute_cdp_cmd('Network.setUserAgentOverride', {
                "userAgent": ua
            })
            logger.debug(f"User-Agent défini: {ua}")

            # Accès initial à Vinted
            logger.info("Navigation vers Vinted...")
            driver.get("https://www.vinted.fr")

            # Attente de chargement générique avec plusieurs indicateurs
            WebDriverWait(driver, 30).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.XPATH, "//body"))
            )
            logger.info("Page principale chargée")

            # Gestion bannière cookies
            accept_cookie_banner(driver)

            # Connexion si nécessaire avec vérification préalable
            if os.getenv("VINTED_USER") and os.getenv("VINTED_PASS") and not is_logged_in(driver):
                if not perform_login(driver):
                    raise RuntimeError("Échec du processus de connexion")

            # Navigation stratégique pour déclencher les cookies
            logger.info("Navigation stratégique pour cookies...")
            driver.get("https://www.vinted.fr/api/v2/catalog/items?per_page=1")
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//pre"))
            )

            # Vérification finale des cookies
            logger.info("Vérification des cookies...")
            if not wait_for_cookies(driver, required_cookies, timeout=45):
                raise ValueError("Cookies requis non présents")

            # Récupération et validation des cookies
            cookies = driver.get_cookies()
            cookies_dict = {c['name']: c['value'] for c in cookies}

            if not required_cookies.issubset(cookies_dict.keys()):
                missing = required_cookies - set(cookies_dict.keys())
                raise ValueError(f"Cookies manquants: {', '.join(missing)}")

            logger.info(f"{len(cookies_dict)} cookies récupérés avec succès")
            return cookies_dict

        except (TimeoutException, WebDriverException, ValueError, RuntimeError) as e:
            logger.error(f"Erreur tentative {attempt}: {type(e).__name__} - {str(e)}")
            if attempt < max_retries:
                retry_delay = min(attempt * 10, 30)  # Backoff exponentiel
                logger.info(f"Nouvelle tentative dans {retry_delay} secondes...")
                time.sleep(retry_delay)
        except Exception as e:
            logger.critical(f"Erreur inattendue: {type(e).__name__} - {str(e)}")
            if attempt < max_retries:
                time.sleep(10)
        finally:
            if driver:
                try:
                    driver.quit()
                except Exception as e:
                    logger.debug(f"Erreur fermeture navigateur: {str(e)}")

    logger.error("Échec après %d tentatives", max_retries)
    return None


if __name__ == "__main__":
    cookies = fetch_vinted_cookies()
    if cookies:
        print("Cookies récupérés avec succès:")
        for name in sorted(cookies.keys()):
            value = cookies[name]
            print(f"{name}: {value[:15]}...")
    else:
        print("Échec de la récupération des cookies")

"""
Version alternative avec Playwright (nécessite pip install playwright):

async def fetch_vinted_cookies_playwright(headless=True):
    from playwright.async_api import async_playwright

    async with async_async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            locale="fr-FR"
        )

        page = await context.new_page()

        try:
            await page.goto("https://www.vinted.fr", wait_until="networkidle")

            # Gestion cookies
            if await page.query_selector("#onetrust-accept-btn-handler"):
                await page.click("#onetrust-accept-btn-handler")

            # Connexion si nécessaire
            if os.getenv("VINTED_USER") and os.getenv("VINTED_PASS"):
                if not await page.query_selector("[data-testid='header__member-menu']"):
                    await page.click("[data-testid='header__login-button']")
                    await page.fill("input#user", os.getenv("VINTED_USER"))
                    await page.fill("input#password", os.getenv("VINTED_PASS"))
                    await page.click("[data-testid='login-form__submit-button']")
                    await page.wait_for_selector("[data-testid='header__member-menu']", timeout=15000)

            # Navigation pour déclencher les cookies
            await page.goto("https://www.vinted.fr/api/v2/catalog/items?per_page=1")

            # Récupération des cookies
            cookies = await context.cookies()
            return {c['name']: c['value'] for c in cookies}

        except Exception as e:
            logger.error(f"Erreur Playwright: {str(e)}")
            return None
        finally:
            await browser.close()
"""
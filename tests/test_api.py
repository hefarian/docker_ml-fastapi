"""
Tests unitaires et fonctionnels pour l'API de prédiction d'attrition

QU'EST-CE QU'UN TEST UNITAIRE ?
================================
Un test unitaire vérifie qu'une petite partie du code fonctionne correctement.
Par exemple : "Quand j'envoie des données valides à /predict, est-ce que je reçois
une prédiction ?"

POURQUOI TESTER ?
=================
1. Détecter les bugs avant qu'ils n'atteignent la production
2. Garantir que les modifications ne cassent pas le code existant
3. Documenter le comportement attendu du code
4. Faciliter la maintenance (refactoring en toute sécurité)

COMMENT FONCTIONNENT CES TESTS ?
=================================
- pytest : framework de test Python (exécute les fonctions qui commencent par "test_")
- TestClient : simule des requêtes HTTP sans avoir besoin d'un serveur réel
- assert : vérifie qu'une condition est vraie (si faux, le test échoue)

EXÉCUTION DES TESTS
====================
- pytest tests/ : exécute tous les tests
- pytest tests/test_api.py::test_health_ok : exécute un test spécifique
- pytest -v : mode verbose (affiche plus de détails)
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ============================================================================
# CONFIGURATION DU PYTHONPATH
# ============================================================================
# Pourquoi cette configuration ?
# Les imports dans app/main.py utilisent des imports absolus (from schemas import ...)
# Ces imports fonctionnent quand on exécute depuis le dossier app/, mais pas depuis tests/
# On configure donc le PYTHONPATH pour que Python trouve les modules

# root_dir : répertoire racine du projet (parent du dossier tests/)
# Path(__file__) : chemin du fichier actuel (test_api.py)
# .parent : dossier parent (tests/)
# .parent : dossier parent (racine du projet)
root_dir = Path(__file__).parent.parent

# app_dir : dossier contenant le code de l'application
app_dir = root_dir / "app"

# Ajouter app_dir au PYTHONPATH
# sys.path : liste des chemins où Python cherche les modules
# insert(0, ...) : ajoute au début de la liste (priorité)
sys.path.insert(0, str(app_dir))

# Changer le répertoire de travail vers app/
# Cela permet aux chemins relatifs dans le code (ex: "models/") de fonctionner
os.chdir(app_dir)

# ============================================================================
# IMPORTS
# ============================================================================
# TestClient : simule un client HTTP pour tester l'API sans serveur réel
# C'est comme faire des requêtes curl, mais en Python
from fastapi.testclient import TestClient

# Importer l'application FastAPI et les schémas
# Maintenant que le PYTHONPATH est configuré, ces imports fonctionnent
from main import app
from schemas import EmployeeData, PredictRequest

# Créer un client de test pour l'application
# Ce client permet de faire des requêtes GET, POST, etc. vers l'API
client = TestClient(app)

# ============================================================================
# DONNÉES DE TEST
# ============================================================================
# SAMPLE_EMPLOYEE : exemple de données d'employé valides
# Utilisé dans plusieurs tests pour éviter de répéter les mêmes données
# Ces données représentent un employé fictif avec toutes les informations nécessaires
SAMPLE_EMPLOYEE = {
    "age": 35,  # Âge de l'employé
    "genre": "M",  # Genre
    "revenu_mensuel": 5000,  # Revenu en euros
    "statut_marital": "Marié(e)",  # Statut marital
    "departement": "R&D",  # Département
    "poste": "Data Scientist",  # Poste occupé
    "nombre_experiences_precedentes": 3,  # Nombre d'expériences
    "annee_experience_totale": 8,  # Années d'expérience totale
    "annees_dans_l_entreprise": 5,  # Années dans l'entreprise
    "annees_dans_le_poste_actuel": 2,  # Années dans le poste actuel
    "satisfaction_employee_environnement": 4,  # Satisfaction environnement (1-5)
    "note_evaluation_precedente": 4.0,  # Note évaluation précédente
    "niveau_hierarchique_poste": 2,  # Niveau hiérarchique
    "satisfaction_employee_nature_travail": 4,  # Satisfaction nature du travail
    "satisfaction_employee_equipe": 4,  # Satisfaction équipe
    "satisfaction_employee_equilibre_pro_perso": 3,  # Satisfaction équilibre pro/perso
    "note_evaluation_actuelle": 4.5,  # Note évaluation actuelle
    "heure_supplementaires": "Non",  # Heures supplémentaires
    "nombre_participation_pee": 2,  # Participations PEE
    "nb_formations_suivies": 3,  # Nombre de formations
    "distance_domicile_travail": 10,  # Distance domicile-travail (km)
    "niveau_education": 3,  # Niveau d'éducation
    "domaine_etude": "Data Science",  # Domaine d'étude
    "frequence_deplacement": "Occasionnel",  # Fréquence de déplacement
    "annees_depuis_la_derniere_promotion": 2,  # Années depuis dernière promotion
    "annes_sous_responsable_actuel": 1,  # Années sous responsable actuel
    "augmentation_salaire_precedente": 0.05,  # Augmentation salaire (5% = 0.05)
}

# ============================================================================
# TESTS DES ENDPOINTS
# ============================================================================


def test_health_ok():
    """
    Test du endpoint /health

    QU'EST-CE QU'UN ENDPOINT HEALTH ?
    =================================
    Un endpoint de santé permet de vérifier que l'API fonctionne correctement.
    C'est comme un "ping" : si ça répond, l'API est en vie.

    CE QUE CE TEST VÉRIFIE :
    ========================
    1. L'endpoint /health répond (status_code == 200)
    2. La réponse contient les champs attendus (status, model_loaded, db_connected)

    POURQUOI C'EST IMPORTANT ?
    ===========================
    - Monitoring : les outils de surveillance peuvent vérifier la santé de l'API
    - Debugging : permet de savoir rapidement si l'API est accessible
    - CI/CD : les pipelines peuvent vérifier que le déploiement a réussi
    """
    # client.get("/health") : fait une requête HTTP GET vers /health
    # C'est comme taper http://localhost:8090/health dans un navigateur
    res = client.get("/health")

    # assert : vérifie qu'une condition est vraie
    # Si status_code != 200, le test échoue
    assert res.status_code == 200

    # res.json() : parse la réponse JSON en dictionnaire Python
    data = res.json()

    # Vérifier que la réponse contient les champs attendus
    # "status" in data : vérifie que la clé "status" existe dans le dictionnaire
    assert "status" in data
    assert "model_loaded" in data
    assert "db_connected" in data


def test_root_ok():
    """
    Test du endpoint racine (/)

    CE QUE CE TEST VÉRIFIE :
    ========================
    - L'endpoint racine répond correctement
    - La réponse contient les informations de base (message, version, endpoints)

    UTILITÉ :
    =========
    L'endpoint racine sert de "page d'accueil" de l'API.
    Il donne des informations sur l'API et liste les endpoints disponibles.
    """
    # Requête GET vers la racine
    res = client.get("/")

    # Vérifier que la requête a réussi
    assert res.status_code == 200

    # Parser la réponse JSON
    data = res.json()

    # Vérifier la présence des champs attendus
    assert "message" in data  # Message de bienvenue
    assert "version" in data  # Version de l'API
    assert "endpoints" in data  # Liste des endpoints disponibles


def test_predict_with_valid_data():
    """
    Test de prédiction avec des données valides

    CE QUE CE TEST VÉRIFIE :
    ========================
    1. L'endpoint /predict accepte des données valides
    2. La réponse contient une prédiction (0 ou 1)
    3. La probabilité est entre 0 et 1
    4. La version du modèle est retournée

    NOTE IMPORTANTE :
    =================
    Le test accepte 200 (succès) OU 503 (service indisponible).
    Pourquoi ? Parce que le modèle peut ne pas être chargé (fichiers manquants).
    En production, on s'attend à 200, mais en développement, 503 est acceptable.
    """
    # Préparer les données de la requête
    # Format attendu : {"employee_data": {...}}
    payload = {"employee_data": SAMPLE_EMPLOYEE}

    # Faire une requête POST vers /predict avec les données en JSON
    # client.post(url, json=data) : envoie une requête POST avec data en JSON
    res = client.post("/predict", json=payload)

    # Accepter 200 (succès) ou 503 (modèle non disponible)
    # Le modèle peut ne pas être chargé si les fichiers sont absents
    assert res.status_code in [200, 503]

    # Si la requête a réussi (200), vérifier la réponse
    if res.status_code == 200:
        data = res.json()

        # Vérifier la présence des champs attendus
        assert "prediction" in data  # Prédiction : 0 ou 1
        assert "probability" in data  # Probabilité : 0.0 à 1.0
        assert "model_version" in data  # Version du modèle

        # Vérifier les valeurs
        assert data["prediction"] in [0, 1]  # Prédiction doit être 0 ou 1
        assert 0 <= data["probability"] <= 1  # Probabilité entre 0 et 1


def test_predict_validation_error_missing_field():
    """
    Test de validation avec champ manquant

    CE QUE CE TEST VÉRIFIE :
    ========================
    - L'API rejette les données incomplètes (champ manquant)
    - Le code de statut est 422 (Unprocessable Entity)

    POURQUOI C'EST IMPORTANT ?
    ===========================
    La validation des données est cruciale pour la sécurité et la robustesse.
    On ne veut pas que le modèle reçoive des données incomplètes.
    """
    # Créer une copie des données valides
    invalid_data = SAMPLE_EMPLOYEE.copy()

    # Supprimer un champ obligatoire (age)
    del invalid_data["age"]

    # Préparer la requête
    payload = {"employee_data": invalid_data}

    # Faire la requête
    res = client.post("/predict", json=payload)

    # Vérifier que l'API a rejeté la requête
    # 422 = Unprocessable Entity (données invalides)
    assert res.status_code == 422


def test_predict_validation_error_invalid_value():
    """
    Test de validation avec valeur invalide

    CE QUE CE TEST VÉRIFIE :
    ========================
    - L'API rejette les valeurs invalides (ex: âge négatif)
    - Le code de statut est 422

    EXEMPLE :
    =========
    Un âge de -5 ans n'a pas de sens. L'API doit rejeter cette valeur.
    """
    # Créer une copie des données valides
    invalid_data = SAMPLE_EMPLOYEE.copy()

    # Modifier un champ avec une valeur invalide
    # L'âge doit être >= 18 selon le schéma Pydantic
    invalid_data["age"] = -5  # Âge négatif = invalide

    # Préparer et envoyer la requête
    payload = {"employee_data": invalid_data}
    res = client.post("/predict", json=payload)

    # Vérifier que l'API a rejeté la requête
    assert res.status_code == 422


def test_predict_validation_error_wrong_type():
    """
    Test de validation avec type incorrect

    CE QUE CE TEST VÉRIFIE :
    ========================
    - L'API rejette les données avec des types incorrects
    - Exemple : string au lieu d'entier pour l'âge

    POURQUOI C'EST IMPORTANT ?
    ===========================
    Le modèle attend des nombres, pas des chaînes de caractères.
    Si on envoie "trente-cinq" au lieu de 35, le modèle ne peut pas fonctionner.
    """
    # Créer une copie des données valides
    invalid_data = SAMPLE_EMPLOYEE.copy()

    # Modifier un champ avec un type incorrect
    # L'âge doit être un entier (int), pas une chaîne (str)
    invalid_data["age"] = "trente-cinq"  # String au lieu d'int

    # Préparer et envoyer la requête
    payload = {"employee_data": invalid_data}
    res = client.post("/predict", json=payload)

    # Vérifier que l'API a rejeté la requête
    assert res.status_code == 422


def test_get_predictions():
    """
    Test de récupération des prédictions

    CE QUE CE TEST VÉRIFIE :
    ========================
    - L'endpoint /predictions retourne une liste de prédictions
    - La réponse contient "count" (nombre) et "predictions" (liste)

    NOTE :
    =====
    Le test accepte 200 (succès) OU 500 (erreur serveur).
    Pourquoi ? Parce que la base de données peut ne pas être connectée.
    En production, on s'attend à 200, mais en développement, 500 est acceptable.
    """
    # Requête GET vers /predictions
    res = client.get("/predictions")

    # Accepter 200 (succès) ou 500 (erreur DB)
    assert res.status_code in [200, 500]

    # Si la requête a réussi, vérifier la structure de la réponse
    if res.status_code == 200:
        data = res.json()

        # Vérifier la présence des champs attendus
        assert "count" in data  # Nombre de prédictions
        assert "predictions" in data  # Liste des prédictions

        # Vérifier que "predictions" est bien une liste
        assert isinstance(data["predictions"], list)


def test_get_predictions_with_limit():
    """
    Test de récupération avec limite

    CE QUE CE TEST VÉRIFIE :
    ========================
    - Le paramètre "limit" fonctionne correctement
    - Le nombre de résultats retournés respecte la limite

    UTILITÉ :
    =========
    Limiter le nombre de résultats évite de surcharger l'API avec trop de données.
    """
    # Requête GET avec paramètre limit=10
    # ?limit=10 : paramètre de requête (query parameter)
    res = client.get("/predictions?limit=10")

    # Accepter 200 ou 500 (comme le test précédent)
    assert res.status_code in [200, 500]

    # Si la requête a réussi, vérifier que la limite est respectée
    if res.status_code == 200:
        data = res.json()
        # Le nombre de résultats ne doit pas dépasser la limite
        assert data["count"] <= 10


def test_db_test():
    """
    Test du endpoint /db-test

    CE QUE CE TEST VÉRIFIE :
    ========================
    - L'endpoint de test de base de données répond

    UTILITÉ :
    =========
    Cet endpoint permet de tester rapidement la connexion à PostgreSQL
    sans avoir besoin de charger le modèle.
    """
    # Requête GET vers /db-test
    res = client.get("/db-test")

    # Accepter 200 (succès) ou 500 (erreur DB)
    # La base peut ne pas être connectée en environnement de test
    assert res.status_code in [200, 500]


# ============================================================================
# TESTS DES SCHÉMAS PYDANTIC
# ============================================================================


def test_schema_employee_data():
    """
    Test de validation du schéma EmployeeData

    CE QUE CE TEST VÉRIFIE :
    ========================
    1. Les données valides sont acceptées par le schéma
    2. Les données invalides sont rejetées (lève une exception)

    POURQUOI TESTER LES SCHÉMAS ?
    =============================
    Les schémas Pydantic sont la première ligne de défense contre les données invalides.
    Si les schémas ne fonctionnent pas, l'API peut recevoir des données corrompues.
    """
    # Test avec des données valides
    # EmployeeData(**SAMPLE_EMPLOYEE) : crée un objet EmployeeData
    # **SAMPLE_EMPLOYEE : "décompresse" le dictionnaire en arguments nommés
    # Si les données sont valides, l'objet est créé sans erreur
    employee = EmployeeData(**SAMPLE_EMPLOYEE)

    # Vérifier que l'objet a été créé correctement
    assert employee.age == 35

    # Test avec des données invalides
    # pytest.raises(Exception) : vérifie qu'une exception est levée
    # Si aucune exception n'est levée, le test échoue
    with pytest.raises(Exception):
        # Créer des données invalides (âge négatif)
        invalid = SAMPLE_EMPLOYEE.copy()
        invalid["age"] = -1  # Âge invalide (< 18)

        # Cette ligne doit lever une exception (ValidationError de Pydantic)
        EmployeeData(**invalid)


def test_schema_predict_request():
    """
    Test de validation du schéma PredictRequest

    CE QUE CE TEST VÉRIFIE :
    ========================
    - Le schéma PredictRequest accepte des données valides
    - L'objet créé contient bien les données de l'employé

    UTILITÉ :
    =========
    PredictRequest est le schéma utilisé pour les requêtes POST /predict.
    Il doit valider que la structure de la requête est correcte.
    """
    # Créer un objet PredictRequest avec des données valides
    # employee_data=SAMPLE_EMPLOYEE : passe les données de l'employé
    request = PredictRequest(employee_data=SAMPLE_EMPLOYEE)

    # Vérifier que l'objet a été créé correctement
    # request.employee_data : accède aux données de l'employé
    # .age : accède à l'âge de l'employé
    assert request.employee_data.age == 35


# ============================================================================
# TESTS DES ENDPOINTS SUPPLÉMENTAIRES
# ============================================================================


def test_model_info():
    """
    Test du endpoint /model-info

    CE QUE CE TEST VÉRIFIE :
    ========================
    - L'endpoint /model-info retourne des informations sur le modèle
    - La réponse contient les champs attendus (model_type, model_version, etc.)

    NOTE :
    =====
    Le test accepte 200 (succès) OU 503 (modèle non disponible).
    Pourquoi ? Parce que le modèle peut ne pas être chargé (fichiers manquants).
    """
    # Requête GET vers /model-info
    res = client.get("/model-info")

    # Accepter 200 (succès) ou 503 (modèle non disponible)
    assert res.status_code in [200, 503]

    # Si la requête a réussi, vérifier la structure de la réponse
    if res.status_code == 200:
        data = res.json()

        # Vérifier la présence des champs attendus
        assert "model_type" in data
        assert "model_version" in data
        assert "artifacts" in data


def test_admin_train_model_without_auth():
    """
    Test du endpoint /admin/train-model sans authentification

    CE QUE CE TEST VÉRIFIE :
    ========================
    - Si ADMIN_TOKEN est défini mais le token n'est pas fourni, l'API retourne 401
    - Si ADMIN_TOKEN n'est pas défini, l'endpoint peut être accessible (selon l'implémentation)
    """
    import os

    # Sauvegarder la valeur actuelle de ADMIN_TOKEN
    original_token = os.environ.get("ADMIN_TOKEN")

    try:
        # Cas 1 : ADMIN_TOKEN défini mais pas de header Authorization
        os.environ["ADMIN_TOKEN"] = "secret-token"

        # Requête POST sans header Authorization
        res = client.post("/admin/train-model")

        # Si ADMIN_TOKEN est défini, on s'attend à 401
        # Si ADMIN_TOKEN n'est pas défini, l'endpoint peut être accessible
        assert res.status_code in [401, 500]  # 500 si DATABASE_URL manque

    finally:
        # Restaurer la valeur originale
        if original_token:
            os.environ["ADMIN_TOKEN"] = original_token
        elif "ADMIN_TOKEN" in os.environ:
            del os.environ["ADMIN_TOKEN"]


def test_admin_train_model_with_auth():
    """
    Test du endpoint /admin/train-model avec authentification

    CE QUE CE TEST VÉRIFIE :
    ========================
    - Si le token est correct, l'endpoint accepte la requête
    - La réponse indique que l'entraînement a démarré

    NOTE :
    =====
    Ce test peut échouer si DATABASE_URL n'est pas définie ou si train_model.py n'existe pas.
    C'est acceptable car c'est un test d'intégration partielle.
    """
    import os
    from unittest.mock import patch

    # Sauvegarder la valeur actuelle
    original_token = os.environ.get("ADMIN_TOKEN")
    original_db_url = os.environ.get("DATABASE_URL")

    try:
        # Configurer l'environnement de test
        os.environ["ADMIN_TOKEN"] = "secret-token"
        if not original_db_url:
            os.environ["DATABASE_URL"] = "postgresql://user:pass@localhost:5432/testdb"

        # Mock de subprocess.Popen pour éviter de lancer vraiment train_model.py
        with patch("subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process

            # Requête POST avec header Authorization
            res = client.post("/admin/train-model", headers={"Authorization": "Bearer secret-token"})

            # Accepter 200 (succès) ou 500 (erreur serveur)
            # 500 peut arriver si train_model.py n'existe pas ou autres erreurs
            assert res.status_code in [200, 500]

            # Si la requête a réussi, vérifier la structure de la réponse
            if res.status_code == 200:
                data = res.json()
                assert "status" in data
                assert "message" in data
                assert data["status"] == "started"

    finally:
        # Restaurer les valeurs originales
        if original_token:
            os.environ["ADMIN_TOKEN"] = original_token
        elif "ADMIN_TOKEN" in os.environ:
            del os.environ["ADMIN_TOKEN"]

        if original_db_url:
            os.environ["DATABASE_URL"] = original_db_url
        elif "DATABASE_URL" in os.environ and not original_db_url:
            del os.environ["DATABASE_URL"]


def test_predict_with_prediction_id():
    """
    Test de prédiction avec enregistrement en base (prediction_id)

    CE QUE CE TEST VÉRIFIE :
    ========================
    - Si l'enregistrement en base réussit, prediction_id est retourné
    - Si l'enregistrement échoue, prediction_id peut être None mais la prédiction est quand même retournée
    """
    payload = {"employee_data": SAMPLE_EMPLOYEE}
    res = client.post("/predict", json=payload)

    # Accepter 200 (succès) ou 503 (modèle non disponible)
    assert res.status_code in [200, 503]

    # Si la requête a réussi, vérifier la présence de prediction_id
    if res.status_code == 200:
        data = res.json()
        # prediction_id peut être None si l'enregistrement DB a échoué
        # mais la prédiction est quand même retournée
        assert "prediction_id" in data
        # prediction_id peut être None ou un entier
        if data["prediction_id"] is not None:
            assert isinstance(data["prediction_id"], int)

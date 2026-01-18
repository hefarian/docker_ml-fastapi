# ============================================================================
# FICHIER : app/main.py
# ============================================================================
#
# QU'EST-CE QUE CE FICHIER ?
# ===========================
# Ce fichier contient l'application FastAPI principale.
# FastAPI est un framework Python moderne pour créer des APIs REST.
#
# QU'EST-CE QU'UNE API REST ?
# ============================
# API = Application Programming Interface (Interface de Programmation)
# REST = Representational State Transfer (architecture web)
# 
# En termes simples : c'est un service web qui accepte des requêtes HTTP
# (GET, POST, etc.) et retourne des réponses JSON.
#
# ENDPOINTS DÉFINIS :
# ===================
# - GET  /health        : Vérifie que l'API fonctionne
# - GET  /              : Page d'accueil avec informations
# - GET  /db-test       : Test de connexion à la base de données
# - POST /predict       : Fait une prédiction d'attrition
# - GET  /predictions   : Récupère les prédictions précédentes
# - GET  /model-info    : Informations sur le modèle
# - GET  /docs          : Documentation interactive (Swagger UI)
# - GET  /redoc         : Documentation alternative (ReDoc)
#
# ============================================================================

# app/main.py

# ============================================================================
# IMPORTS
# ============================================================================
from fastapi import FastAPI, HTTPException, Header
from schemas import PredictRequest, PredictResponse, HealthResponse
from model import get_model
from database import get_database
from settings import settings
from pathlib import Path
import joblib
import traceback  # importé mais non utilisé dans ce fichier ; utile si tu veux logger les stack traces

# ============================================================================
# INSTANCIATION DE L'APPLICATION FASTAPI
# ============================================================================
# Instanciation de l'application FastAPI avec métadonnées
# - title / description : visibles dans la doc auto (Swagger /docs et ReDoc /redoc)
# - version : pratique pour le suivi et la compatibilité clients
app = FastAPI(
    title="API de Prédiction d'Attrition",
    description="API pour prédire l'attrition des employés avec un modèle XGBoost",
    version="1.0.0"
)

# ============================================================================
# ENDPOINT : /health (Vérification de santé)
# ============================================================================
@app.get("/health", response_model=HealthResponse)
def health():
    """
    Endpoint de santé de l'API.
    
    QU'EST-CE QU'UN ENDPOINT DE SANTÉ ?
    ====================================
    Un endpoint de santé permet de vérifier que l'API et ses dépendances
    fonctionnent correctement. C'est comme un "ping" pour l'API.
    
    UTILISATIONS :
    ==============
    - Monitoring : les outils de surveillance peuvent vérifier la santé automatiquement
    - CI/CD : vérifier que le déploiement a réussi
    - Debugging : diagnostiquer rapidement les problèmes
    
    Objectifs :
    - Vérifier la disponibilité des dépendances critiques :
        * Chargement du modèle (artefacts présents, pas corrompus).
        * Connexion à la base de données (ping basique).
    - Retourner un statut global "ok" si tout est disponible, sinon "degraded".
    
    Notes d'implémentation :
    - On encapsule chaque vérification dans un try/except pour ne pas court-circuiter l'autre.
    - En prod, on pourrait :
        * tracer l'exception (logs) pour diagnostic.
        * différencier "degraded" par service (ex: model_only, db_only).
    
    RETOUR :
    ========
    HealthResponse avec :
    - status : "ok" ou "degraded"
    - model_loaded : True si le modèle est chargé
    - db_connected : True si la base est accessible
    """
    # Variables pour suivre l'état de chaque dépendance
    model_loaded = False
    db_connected = False

    # ========================================================================
    # VÉRIFICATION 1 : Modèle chargé ?
    # ========================================================================
    try:
        # get_model() : charge le modèle (singleton, mis en cache)
        # Si le modèle n'est pas disponible, une exception est levée
        _ = get_model()     # tente de charger ou récupérer le singleton modèle
        model_loaded = True
    except Exception:
        # Ici on ignore l'exception pour ne pas faire échouer l'endpoint health entièrement.
        # En pratique, logguer l'erreur (ex: logger.exception("Model load failed")).
        # Si le modèle n'est pas chargé, model_loaded reste False
        pass

    # ========================================================================
    # VÉRIFICATION 2 : Base de données accessible ?
    # ========================================================================
    try:
        # get_database() : récupère l'instance Database (singleton via lru_cache)
        db = get_database()
        # test_connection() : exécute "SELECT 1" pour vérifier la connexion
        db_connected = db.test_connection()  # SELECT 1 ; True si OK
    except Exception:
        # Idem : ne pas faire échouer /health, mais consigner l'état dégradé.
        # Si la base n'est pas accessible, db_connected reste False
        pass

    # ========================================================================
    # CALCUL DU STATUT GLOBAL
    # ========================================================================
    # Statut global :
    # - "ok" : modèle chargé ET DB accessible
    # - "degraded" : au moins une des deux dépendances indisponible
    status = "ok" if model_loaded and db_connected else "degraded"

    # La réponse est validée/sérialisée par le modèle Pydantic HealthResponse
    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        db_connected=db_connected
    )

# ============================================================================
# ENDPOINT : / (Page d'accueil)
# ============================================================================
@app.get("/")
def read_root():
    """
    Endpoint racine (landing endpoint) avec informations et liens utiles.
    
    QU'EST-CE QU'UN ENDPOINT RACINE ?
    ==================================
    L'endpoint "/" est la page d'accueil de l'API.
    Quand quelqu'un accède à http://votre-api.com/, c'est cette fonction qui répond.
    
    UTILITÉ :
    =========
    - Donne une vue d'ensemble de l'API
    - Liste les endpoints disponibles
    - Fournit des informations de base (nom, version)
    
    Retourne un petit manifeste JSON :
    - message : nom de l'API
    - version : version de l'API
    - endpoints : chemins des endpoints principaux (auto-documentation utile)
    """
    return {
        "message": "API de Prédiction d'Attrition",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",           # Vérification de santé
            "predict": "/predict",         # Prédiction d'attrition
            "docs": "/docs",               # Swagger UI auto (générée par FastAPI)
            "predictions": "/predictions", # Historique des prédictions
            "model_info": "/model-info"    # Informations sur le modèle
        }
    }

# ============================================================================
# ENDPOINT : /db-test (Test de connexion à la base)
# ============================================================================
@app.get("/db-test")
def db_test():
    """
    Endpoint simple pour tester la connectivité à la base de données.
    
    QU'EST-CE QUE CE ENDPOINT ?
    ============================
    Cet endpoint teste uniquement la connexion à PostgreSQL.
    Contrairement à /health, il ne teste pas le modèle.
    
    UTILITÉ :
    =========
    - Diagnostiquer rapidement un problème de DB indépendamment du modèle
    - Vérifier que les credentials sont corrects
    - Tester la connectivité réseau
    
    Comportement :
    - Si la DB répond à SELECT 1, renvoie {"status": "ok", ...}
    - Sinon lève une HTTPException 500.
    
    Intérêt :
    - Utile pour diagnostiquer rapidement un problème de DB indépendamment du modèle.
    """
    try:
        # Récupérer l'instance Database
        db = get_database()
        
        # Tester la connexion
        if db.test_connection():
            # Connexion OK
            return {"status": "ok", "message": "Connexion DB réussie"}
        else:
            # DB jointe mais test_connection() retourne False
            # (par ex. si SELECT 1 échoue sans lever d'exception)
            raise HTTPException(status_code=500, detail="Échec de connexion à la DB")
    except Exception as e:
        # En cas d'erreur inattendue (connexion impossible, pool non initialisable, etc.)
        # on remonte une 500 avec le message contextuel.
        # En prod, préférer un message générique côté client et logger l'exception côté serveur.
        raise HTTPException(status_code=500, detail=f"Erreur DB: {e}")

# ============================================================================
# ENDPOINT : /predict (Prédiction d'attrition)
# ============================================================================
@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    """
    Prédit l'attrition pour un employé (modèle XGBoost).
    
    QU'EST-CE QUE CET ENDPOINT ?
    ============================
    C'est l'endpoint principal de l'API. Il prend les données d'un employé
    et retourne une prédiction : l'employé va-t-il quitter l'entreprise ?
    
    COMMENT ÇA MARCHE ?
    ===================
    1. Le client envoie une requête POST avec les données de l'employé
    2. FastAPI valide automatiquement les données avec Pydantic
    3. Le modèle fait une prédiction
    4. La prédiction est enregistrée en base de données
    5. La réponse est retournée au client
    
    Entrée:
    - payload: PredictRequest (Pydantic), inclut employee_data (features d'entrée)
    
    Sortie (PredictResponse):
    - prediction (int): 0 = reste, 1 = quitte
    - probability (float): proba associée à la classe prédite
    - model_version (str): version du modèle (pour traçabilité)
    - prediction_id (int|None): identifiant de la ligne insérée en DB si l'enregistrement a réussi
    
    Stratégie :
    1) Charger/récupérer le modèle via get_model() (pattern singleton/cache).
    2) Appeler model.predict(...) avec les données du payload (types validés par Pydantic).
    3) Tenter d'enregistrer la prédiction en base :
       - En cas d'échec DB, on log et on n'empêche pas de renvoyer la prédiction au client.
    4) Gérer distinctement les erreurs :
       - FileNotFoundError -> artefacts de modèle manquants -> 503 (service indisponible)
       - Autres -> 400 (erreur de requête/prédiction)
    """
    try:
        # ====================================================================
        # ÉTAPE 1 : CHARGER LE MODÈLE
        # ====================================================================
        # 1) Charger le modèle (Booster XGBoost + encoders)
        # get_model() : récupère le modèle (singleton, mis en cache)
        # Si le modèle n'est pas disponible, FileNotFoundError est levée
        model = get_model()

        # ====================================================================
        # ÉTAPE 2 : FAIRE LA PRÉDICTION
        # ====================================================================
        # 2) Faire la prédiction — la méthode du modèle renvoie (prediction, probability)
        #    Remarque : la signature exacte dépend de ton wrapper modèle ; ici elle est supposée.
        # payload.employee_data : données de l'employé (validées par Pydantic)
        # model.predict() : transforme les données et fait la prédiction
        prediction, probability = model.predict(payload.employee_data)

        # ====================================================================
        # ÉTAPE 3 : ENREGISTRER EN BASE DE DONNÉES
        # ====================================================================
        # 3) Enregistrer dans la base de données (non bloquant pour la réponse au client)
        # On enregistre même si ça échoue (on retourne quand même la prédiction)
        db = get_database()
        prediction_id = None
        try:
            # .model_dump() : méthode Pydantic v2 pour obtenir un dict natif (équivalent de .dict() en v1)
            # On enregistre :
            # - input_data : les données d'entrée (pour traçabilité)
            # - prediction : le résultat (0 ou 1)
            # - probability : la probabilité (0.0 à 1.0)
            # - model_version : la version du modèle (pour savoir quel modèle a fait la prédiction)
            prediction_id = db.save_prediction(
                input_data=payload.employee_data.model_dump(),
                prediction=prediction,
                probability=probability,
                model_version=settings.MODEL_VERSION  # version récupérée depuis la config
            )
        except Exception as db_error:
            # Choix UX : on ne bloque pas la route si l'enregistrement échoue (retourne quand même la prédiction).
            # En prod, mieux : logger avec niveau WARNING/ERROR + trace (logger.exception) au lieu de print.
            print(f"Erreur lors de l'enregistrement en DB: {db_error}")

        # ====================================================================
        # ÉTAPE 4 : RETOURNER LA RÉPONSE
        # ====================================================================
        # 4) Réponse validée par PredictResponse (assure types et sérialisation)
        return PredictResponse(
            prediction=prediction,
            probability=float(probability),  # cast explicite -> JSON (float natif)
            model_version=settings.MODEL_VERSION,
            prediction_id=prediction_id
        )
    except FileNotFoundError as e:
        # ====================================================================
        # ERREUR : MODÈLE NON DISPONIBLE
        # ====================================================================
        # Cas typique : artefacts du modèle absents (ex: après déploiement sans entraînement)
        # 503 = Service Unavailable, invite à exécuter le pipeline d'entraînement
        raise HTTPException(
            status_code=503,
            detail=f"Modèle non disponible. Veuillez entraîner le modèle d'abord: {e}"
        )
    except Exception as e:
        # ====================================================================
        # ERREUR GÉNÉRIQUE
        # ====================================================================
        # Erreur générique de prédiction (données non conformes, bug dans le wrapper, etc.)
        # 400 = Bad Request : pour garder la distinction avec erreur serveur interne.
        # En prod : logger l'exception avec stack trace pour debug.
        raise HTTPException(
            status_code=400,
            detail=f"Erreur de prédiction: {str(e)}"
        )

# ============================================================================
# ENDPOINT : /predictions (Historique des prédictions)
# ============================================================================
@app.get("/predictions")
def get_predictions(limit: int = 100):
    """
    Récupère les dernières prédictions enregistrées.
    
    QU'EST-CE QUE CET ENDPOINT ?
    ============================
    Cet endpoint permet de consulter l'historique des prédictions faites par l'API.
    Utile pour :
    - Analyser les prédictions passées
    - Déboguer des problèmes
    - Auditer l'utilisation de l'API
    
    Paramètres (query) :
    - limit (int) : nombre maximum de lignes à retourner (défaut 100).
    
    Exemple d'utilisation :
    - GET /predictions : retourne les 100 dernières prédictions
    - GET /predictions?limit=10 : retourne les 10 dernières prédictions
    
    Comportement :
    - Lit depuis la DB via db.get_predictions(limit).
    - Sérialise proprement les champs :
        * probability -> float (compat JSON)
        * created_at -> ISO 8601 string (datetime -> .isoformat())
    - Renvoie un objet JSON contenant le nombre et la liste des prédictions.
    
    Remarques :
    - Si le volume augmente, pense à une pagination (keyset pagination) et des index sur created_at.
    """
    try:
        # Récupérer l'instance Database
        db = get_database()
        
        # Récupérer les prédictions depuis la base
        predictions = db.get_predictions(limit=limit)

        # Normalisation/sérialisation des champs pour la réponse JSON
        return {
            "count": len(predictions),  # Nombre de prédictions retournées
            "predictions": [
                {
                    "id": p["id"],                                    # ID unique de la prédiction
                    "input_data": p["input_data"],                    # dict JSON (si colonne JSONB en DB)
                    "prediction": p["prediction"],                    # Prédiction (0 ou 1)
                    "probability": float(p["probability"]),           # cast explicite -> JSON-friendly
                    "model_version": p["model_version"],              # Version du modèle utilisé
                    # created_at : datetime -> ISO8601 ; protège si la clé n'existe pas
                    "created_at": p["created_at"].isoformat() if p.get("created_at") else None
                }
                for p in predictions  # Liste en compréhension : crée un dict pour chaque prédiction
            ]
        }
    except Exception as e:
        # Erreur côté base (connexion, requête, sérialisation...)
        # 500 = erreur serveur ; le message inclut l'exception (utile côté dev, prudence en prod).
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération: {e}")


# ============================================================================
# ENDPOINT : /model-info (Informations sur le modèle)
# ============================================================================
@app.get("/model-info")
def model_info():
    """
    Retourne des métadonnées sur le modèle XGBoost et ses artefacts.
    
    QU'EST-CE QUE CET ENDPOINT ?
    ============================
    Cet endpoint fournit des informations techniques sur le modèle :
    - Type de modèle
    - Version
    - Nombre de features
    - Hyperparamètres utilisés
    - Chemins des fichiers
    
    UTILITÉ :
    =========
    - Debugging : vérifier quel modèle est chargé
    - Audit : connaître les hyperparamètres utilisés
    - Monitoring : vérifier la version du modèle
    
    Contenu :
    - type de modèle, version
    - nombre de features et échantillon des noms
    - best_iteration / best_score (si early stopping)
    - chemins des artefacts (booster JSON, encoders, feature_names, best_params)
    - meilleurs hyperparamètres (Optuna) si disponibles
    
    Stratégie :
    1) Charger le modèle (mêmes artefacts que pour la prédiction).
    2) Déduire le dossier des modèles à partir de settings.MODEL_PATH (supposé pointer un artefact).
    3) Extraire les attributs du booster (best_iteration, best_score) si présents.
    4) Tenter de charger les "best_params" depuis un fichier joblib ; en cas d'échec, ne pas bloquer.
    """
    try:
        # ====================================================================
        # ÉTAPE 1 : CHARGER LE MODÈLE
        # ====================================================================
        # 1) Charger le modèle (Booster + encoders + feature_names)
        model = get_model()

        # ====================================================================
        # ÉTAPE 2 : DÉTERMINER LES CHEMINS DES ARTEFACTS
        # ====================================================================
        # 2) Chemin des artefacts : on part du chemin du modèle (paramétré) et on remonte au dossier parent
        models_dir = Path(settings.MODEL_PATH).parent
        best_params_path = models_dir / "xgb_best_params.joblib"

        # ====================================================================
        # ÉTAPE 3 : EXTRAIRE LES INFORMATIONS DU BOOSTER
        # ====================================================================
        # 3) Extraire infos du booster ; getattr(...) pour éviter AttributeError si champs absents
        # best_iteration : meilleur nombre d'itérations trouvé par early stopping
        best_iter = getattr(model.booster, "best_iteration", None)
        # best_score : meilleur score AUC sur le jeu de validation
        best_score = getattr(model.booster, "best_score", None)

        # ====================================================================
        # ÉTAPE 4 : CONSTRUIRE LE DICTIONNAIRE D'INFORMATIONS
        # ====================================================================
        # Construction du bloc d'informations
        info = {
            "model_type": "xgboost.Booster",                    # Type de modèle
            "model_version": settings.MODEL_VERSION,            # Version du modèle
            "n_features": len(model.feature_names) if hasattr(model, "feature_names") else None,  # Nombre de features
            "features_sample": model.feature_names[:10] if hasattr(model, "feature_names") else None,  # Échantillon des noms
            "best_iteration": int(best_iter) if best_iter is not None else None,  # Meilleur nombre d'itérations
            "best_score_valid_auc": float(best_score) if best_score is not None else None,  # Meilleur AUC
            "artifacts": {
                # On expose des chemins absolus pour faciliter le diagnostic en environnement serveur
                "booster_json": str((models_dir / "xgb_booster.json").resolve()),
                "onehot_encoder": str((models_dir / "onehot_encoder.joblib").resolve()),
                "ordinal_encoder": str((models_dir / "ordinal_encoder.joblib").resolve()),
                "feature_names": str((models_dir / "feature_names.joblib").resolve()),
                "best_params": str(best_params_path.resolve()),
            },
        }

        # ====================================================================
        # ÉTAPE 5 : CHARGER LES HYPERPARAMÈTRES OPTUNA
        # ====================================================================
        # 4) Charger les meilleurs hyperparamètres Optuna si le fichier existe
        if best_params_path.exists():
            try:
                # Charger le fichier joblib contenant les hyperparamètres
                best_art = joblib.load(best_params_path)
                # On sécurise l'accès via .get(); conversion de types pour JSON
                info["best_params"] = best_art.get("best_params")
                info["best_num_boost_round"] = int(best_art.get("best_num_boost_round")) \
                    if best_art.get("best_num_boost_round") is not None else None
            except Exception as e:
                # Ne bloque pas l'endpoint ; on informe seulement du problème de lecture
                info["best_params_error"] = f"Impossible de charger xgb_best_params.joblib: {e}"
                info["best_params"] = None
                info["best_num_boost_round"] = None
        else:
            # Fichier absent -> ces infos ne sont simplement pas disponibles
            info["best_params"] = None
            info["best_num_boost_round"] = None

        return info

    except FileNotFoundError as e:
        # Artefacts modèle manquants (ex: déploiement avant entraînement)
        raise HTTPException(
            status_code=503,
            detail=f"Modèle non disponible. Veuillez entraîner le modèle d'abord: {e}"
        )
    except Exception as e:
        # Problème générique d'accès/lecture/attributs du modèle ou des artefacts
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération des informations modèle: {str(e)}"
        )


# ============================================================================
# ENDPOINT : /admin/train-model (Déclenchement de l'entraînement)
# ============================================================================
@app.post("/admin/train-model")
async def trigger_training(authorization: str = Header(None)):
    """
    Endpoint pour déclencher l'entraînement du modèle sur Render.
    
    QU'EST-CE QUE CET ENDPOINT ?
    ============================
    Cet endpoint permet de lancer le script train_model.py directement depuis Render.
    Utile pour réentraîner le modèle sans avoir à se connecter au Shell Render.
    
    AUTHENTIFICATION :
    ==================
    Envoyez le token dans le header Authorization :
    Authorization: Bearer votre-token-secret
    
    Le token doit être défini dans les variables d'environnement Render :
    ADMIN_TOKEN=votre-token-secret
    
    COMMENT ÇA MARCHE ?
    ===================
    1. Vérifie l'authentification (token)
    2. Lance train_model.py en arrière-plan
    3. Retourne immédiatement (ne bloque pas l'API)
    4. L'entraînement s'exécute en arrière-plan
    
    NOTE IMPORTANTE :
    =================
    - L'entraînement peut prendre plusieurs minutes (10-30 min selon OPTUNA_TRIALS)
    - Vérifiez les logs Render pour suivre la progression
    - Le modèle sera sauvegardé dans models/ une fois terminé
    - En production, utilisez une queue (Celery, RQ) pour éviter les timeouts
    
    EXEMPLE D'UTILISATION :
    =======================
    curl -X POST https://votre-api.render.com/admin/train-model \
      -H "Authorization: Bearer votre-token-secret"
    """
    import subprocess
    import os
    
    # ========================================================================
    # VÉRIFICATION DE L'AUTHENTIFICATION
    # ========================================================================
    # Récupérer le token depuis les variables d'environnement
    ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")
    
    # Si ADMIN_TOKEN est défini, vérifier l'authentification
    if ADMIN_TOKEN:
        # Vérifier que le header Authorization est présent
        if not authorization:
            raise HTTPException(
                status_code=401,
                detail="Token d'authentification manquant. Envoyez-le dans le header Authorization: Bearer <token>"
            )
        
        # Vérifier que le token correspond
        expected_auth = f"Bearer {ADMIN_TOKEN}"
        if authorization != expected_auth:
            raise HTTPException(
                status_code=401,
                detail="Token d'authentification invalide"
            )
    
    # ========================================================================
    # VÉRIFICATION DES PRÉREQUIS
    # ========================================================================
    # Vérifier que DATABASE_URL est définie
    if not os.getenv("DATABASE_URL"):
        raise HTTPException(
            status_code=500,
            detail="DATABASE_URL non définie. Configurez-la dans les variables d'environnement Render."
        )
    
    # Vérifier que train_model.py existe
    train_script = Path("train_model.py")
    if not train_script.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Fichier train_model.py introuvable dans {Path.cwd()}"
        )
    
    # ========================================================================
    # LANCEMENT DE L'ENTRAÎNEMENT
    # ========================================================================
    try:
        # Créer le dossier models/ s'il n'existe pas
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Exécuter le script d'entraînement en arrière-plan
        # subprocess.Popen : lance un processus sans attendre sa fin
        # stdout/stderr : rediriger vers des pipes (pour les logs)
        process = subprocess.Popen(
            ["python", "train_model.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="/app"  # Répertoire de travail (Render utilise /app)
        )
        
        return {
            "status": "started",
            "message": "Entraînement démarré avec succès",
            "pid": process.pid,
            "note": "L'entraînement s'exécute en arrière-plan. Vérifiez les logs Render pour suivre la progression.",
            "estimated_time": "10-30 minutes selon OPTUNA_TRIALS",
            "logs": "Consultez les logs Render pour voir la progression en temps réel"
        }
    except Exception as e:
        # En cas d'erreur lors du lancement
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du démarrage de l'entraînement: {str(e)}"
        )

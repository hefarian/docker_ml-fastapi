
# app/main.py

from fastapi import FastAPI, HTTPException
from schemas import PredictRequest, PredictResponse, HealthResponse
from model import get_model
from database import get_database
from settings import settings
from pathlib import Path
import joblib
import traceback  # importé mais non utilisé dans ce fichier ; utile si tu veux logger les stack traces

# Instanciation de l'application FastAPI avec métadonnées
# - title / description : visibles dans la doc auto (Swagger /docs et ReDoc /redoc)
# - version : pratique pour le suivi et la compatibilité clients
app = FastAPI(
    title="API de Prédiction d'Attrition",
    description="API pour prédire l'attrition des employés avec un modèle XGBoost",
    version="1.0.0"
)

@app.get("/health", response_model=HealthResponse)
def health():
    """Endpoint de santé de l'API.

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
    """
    model_loaded = False
    db_connected = False

    try:
        _ = get_model()     # tente de charger ou récupérer le singleton modèle
        model_loaded = True
    except Exception:
        # Ici on ignore l'exception pour ne pas faire échouer l'endpoint health entièrement.
        # En pratique, logguer l'erreur (ex: logger.exception("Model load failed")).
        pass

    try:
        db = get_database()  # récupère l'instance Database (singleton via lru_cache)
        db_connected = db.test_connection()  # SELECT 1 ; True si OK
    except Exception:
        # Idem : ne pas faire échouer /health, mais consigner l'état dégradé.
        pass

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

@app.get("/")
def read_root():
    """Endpoint racine (landing endpoint) avec informations et liens utiles.

    Retourne un petit manifeste JSON :
    - message : nom de l'API
    - version : version de l'API
    - endpoints : chemins des endpoints principaux (auto-documentation utile)
    """
    return {
        "message": "API de Prédiction d'Attrition",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",           # Swagger UI auto (générée par FastAPI)
            "predictions": "/predictions",
            "model_info": "/model-info"
        }
    }

@app.get("/db-test")
def db_test():
    """Endpoint simple pour tester la connectivité à la base de données.

    Comportement :
    - Si la DB répond à SELECT 1, renvoie {"status": "ok", ...}
    - Sinon lève une HTTPException 500.

    Intérêt :
    - Utile pour diagnostiquer rapidement un problème de DB indépendamment du modèle.
    """
    try:
        db = get_database()
        if db.test_connection():
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

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    """
    Prédit l'attrition pour un employé (modèle XGBoost).

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
        # 1) Charger le modèle (Booster XGBoost + encoders)
        model = get_model()

        # 2) Faire la prédiction — la méthode du modèle renvoie (prediction, probability)
        #    Remarque : la signature exacte dépend de ton wrapper modèle ; ici elle est supposée.
        prediction, probability = model.predict(payload.employee_data)

        # 3) Enregistrer dans la base de données (non bloquant pour la réponse au client)
        db = get_database()
        prediction_id = None
        try:
            # .model_dump() : méthode Pydantic v2 pour obtenir un dict natif (équivalent de .dict() en v1)
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

        # 4) Réponse validée par PredictResponse (assure types et sérialisation)
        return PredictResponse(
            prediction=prediction,
            probability=float(probability),  # cast explicite -> JSON (float natif)
            model_version=settings.MODEL_VERSION,
            prediction_id=prediction_id
        )
    except FileNotFoundError as e:
        # Cas typique : artefacts du modèle absents (ex: après déploiement sans entraînement)
        # 503 = Service Unavailable, invite à exécuter le pipeline d'entraînement
        raise HTTPException(
            status_code=503,
            detail=f"Modèle non disponible. Veuillez entraîner le modèle d'abord: {e}"
        )
    except Exception as e:
        # Erreur générique de prédiction (données non conformes, bug dans le wrapper, etc.)
        # 400 = Bad Request : pour garder la distinction avec erreur serveur interne.
        # En prod : logger l'exception avec stack trace pour debug.
        raise HTTPException(
            status_code=400,
            detail=f"Erreur de prédiction: {str(e)}"
        )

@app.get("/predictions")
def get_predictions(limit: int = 100):
    """
    Récupère les dernières prédictions enregistrées.

    Paramètres (query) :
    - limit (int) : nombre maximum de lignes à retourner (défaut 100).

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
        db = get_database()
        predictions = db.get_predictions(limit=limit)

        # Normalisation/sérialisation des champs pour la réponse JSON
        return {
            "count": len(predictions),
            "predictions": [
                {
                    "id": p["id"],
                    "input_data": p["input_data"],  # dict JSON (si colonne JSONB en DB)
                    "prediction": p["prediction"],
                    "probability": float(p["probability"]),  # cast explicite -> JSON-friendly
                    "model_version": p["model_version"],
                    # created_at : datetime -> ISO8601 ; protège si la clé n'existe pas
                    "created_at": p["created_at"].isoformat() if p.get("created_at") else None
                }
                for p in predictions
            ]
        }
    except Exception as e:
        # Erreur côté base (connexion, requête, sérialisation...)
        # 500 = erreur serveur ; le message inclut l'exception (utile côté dev, prudence en prod).
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération: {e}")


@app.get("/model-info")
def model_info():
    """
    Retourne des métadonnées sur le modèle XGBoost et ses artefacts.

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
        # 1) Charger le modèle (Booster + encoders + feature_names)
        model = get_model()

        # 2) Chemin des artefacts : on part du chemin du modèle (paramétré) et on remonte au dossier parent
        models_dir = Path(settings.MODEL_PATH).parent
        best_params_path = models_dir / "xgb_best_params.joblib"

        # 3) Extraire infos du booster ; getattr(...) pour éviter AttributeError si champs absents
        best_iter = getattr(model.booster, "best_iteration", None)
        best_score = getattr(model.booster, "best_score", None)

        # Construction du bloc d'informations
        info = {
            "model_type": "xgboost.Booster",
            "model_version": settings.MODEL_VERSION,
            "n_features": len(model.feature_names) if hasattr(model, "feature_names") else None,
            "features_sample": model.feature_names[:10] if hasattr(model, "feature_names") else None,
            "best_iteration": int(best_iter) if best_iter is not None else None,
            "best_score_valid_auc": float(best_score) if best_score is not None else None,
            "artifacts": {
                # On expose des chemins absolus pour faciliter le diagnostic en environnement serveur
                "booster_json": str((models_dir / "xgb_booster.json").resolve()),
                "onehot_encoder": str((models_dir / "onehot_encoder.joblib").resolve()),
                "ordinal_encoder": str((models_dir / "ordinal_encoder.joblib").resolve()),
                "feature_names": str((models_dir / "feature_names.joblib").resolve()),
                "best_params": str(best_params_path.resolve()),
            },
        }

        # 4) Charger les meilleurs hyperparamètres Optuna si le fichier existe
        if best_params_path.exists():
            try:
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

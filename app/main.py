
# app/main.py
from fastapi import FastAPI, HTTPException
from schemas import PredictRequest, PredictResponse, HealthResponse
from model import get_model
from database import get_database
from settings import settings
from pathlib import Path
import joblib
import traceback

app = FastAPI(
    title="API de Prédiction d'Attrition",
    description="API pour prédire l'attrition des employés avec un modèle XGBoost",
    version="1.0.0"
)

@app.get("/health", response_model=HealthResponse)
def health():
    """Endpoint de santé de l'API"""
    model_loaded = False
    db_connected = False

    try:
        _ = get_model()
        model_loaded = True
    except Exception:
        pass

    try:
        db = get_database()
        db_connected = db.test_connection()
    except Exception:
        pass

    status = "ok" if model_loaded and db_connected else "degraded"

    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        db_connected=db_connected
    )

@app.get("/")
def read_root():
    """Endpoint racine avec documentation"""
    return {
        "message": "API de Prédiction d'Attrition",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
            "predictions": "/predictions",
            "model_info": "/model-info"
        }
    }

@app.get("/db-test")
def db_test():
    """Test de connexion à la base de données"""
    try:
        db = get_database()
        if db.test_connection():
            return {"status": "ok", "message": "Connexion DB réussie"}
        else:
            raise HTTPException(status_code=500, detail="Échec de connexion à la DB")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur DB: {e}")

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    """
    Prédit l'attrition pour un employé (modèle XGBoost)

    - **employee_data**: Données de l'employé (âge, revenu, département, etc.)
    - Retourne la prédiction (0 = reste, 1 = quitte) et la probabilité associée
    """
    try:
        # Charger le modèle (Booster XGBoost + encoders)
        model = get_model()

        # Faire la prédiction
        prediction, probability = model.predict(payload.employee_data)

        # Enregistrer dans la base de données
        db = get_database()
        prediction_id = None
        try:
            prediction_id = db.save_prediction(
                input_data=payload.employee_data.model_dump(),
                prediction=prediction,
                probability=probability,
                model_version=settings.MODEL_VERSION
            )
        except Exception as db_error:
            # Log l'erreur mais ne bloque pas la prédiction
            print(f"Erreur lors de l'enregistrement en DB: {db_error}")

        return PredictResponse(
            prediction=prediction,
            probability=float(probability),
            model_version=settings.MODEL_VERSION,
            prediction_id=prediction_id
        )
    except FileNotFoundError as e:
        # Artefacts du modèle manquants (exécuter le script d'entraînement)
        raise HTTPException(
            status_code=503,
            detail=f"Modèle non disponible. Veuillez entraîner le modèle d'abord: {e}"
        )
    except Exception as e:
        # Toute autre erreur côté prédiction
        raise HTTPException(
            status_code=400,
            detail=f"Erreur de prédiction: {str(e)}"
        )

@app.get("/predictions")
def get_predictions(limit: int = 100):
    """
    Récupère les dernières prédictions enregistrées

    - **limit**: Nombre maximum de prédictions à retourner (défaut: 100)
    """
    try:
        db = get_database()
        predictions = db.get_predictions(limit=limit)
        return {
            "count": len(predictions),
            "predictions": [
                {
                    "id": p["id"],
                    "input_data": p["input_data"],
                    "prediction": p["prediction"],
                    "probability": float(p["probability"]),
                    "model_version": p["model_version"],
                    "created_at": p["created_at"].isoformat() if p.get("created_at") else None
                }
                for p in predictions
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération: {e}")


@app.get("/model-info")
def model_info():
    """
    Retourne des métadonnées du modèle XGBoost :
    - type, version
    - nombre de features, échantillon des noms
    - best_iteration / best_score (si early stopping)
    - hyperparamètres Optuna (si présents)
    """
    try:
        # 1) Charger le modèle (Booster + encoders + feature_names)
        model = get_model()

        # 2) Chemin des artefacts (même logique que get_model)
        models_dir = Path(settings.MODEL_PATH).parent
        best_params_path = models_dir / "xgb_best_params.joblib"

        # 3) Extraire infos du booster
        best_iter = getattr(model.booster, "best_iteration", None)
        best_score = getattr(model.booster, "best_score", None)

        info = {
            "model_type": "xgboost.Booster",
            "model_version": settings.MODEL_VERSION,
            "n_features": len(model.feature_names) if hasattr(model, "feature_names") else None,
            "features_sample": model.feature_names[:10] if hasattr(model, "feature_names") else None,
            "best_iteration": int(best_iter) if best_iter is not None else None,
            "best_score_valid_auc": float(best_score) if best_score is not None else None,
            "artifacts": {
                "booster_json": str((models_dir / "xgb_booster.json").resolve()),
                "onehot_encoder": str((models_dir / "onehot_encoder.joblib").resolve()),
                "ordinal_encoder": str((models_dir / "ordinal_encoder.joblib").resolve()),
                "feature_names": str((models_dir / "feature_names.joblib").resolve()),
                "best_params": str(best_params_path.resolve()),
            },
        }

        # 4) Charger les meilleurs hyperparamètres Optuna si dispo
        if best_params_path.exists():
            try:
                best_art = joblib.load(best_params_path)
                info["best_params"] = best_art.get("best_params")
                info["best_num_boost_round"] = int(best_art.get("best_num_boost_round")) \
                    if best_art.get("best_num_boost_round") is not None else None
            except Exception as e:
                # Ne bloque pas, juste tracer l’erreur
                info["best_params_error"] = f"Impossible de charger xgb_best_params.joblib: {e}"
                info["best_params"] = None
                info["best_num_boost_round"] = None
        else:
            info["best_params"] = None
            info["best_num_boost_round"] = None

        return info

    except FileNotFoundError as e:
        # Artefacts du modèle manquants (exécuter le script d'entraînement)
        raise HTTPException(
            status_code=503,
            detail=f"Modèle non disponible. Veuillez entraîner le modèle d'abord: {e}"
        )
    except Exception as e:
        # Toute autre erreur côté introspection modèle
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération des informations modèle: {str(e)}"
        )

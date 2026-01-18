# app/settings.py
import os
from pathlib import Path

class Settings:
    MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
    MODEL_PATH = os.getenv("MODEL_PATH", "models/logistic_regression_model.joblib")
    MODELS_DIR = Path("models")

settings = Settings()

# app/settings.py
"""
Module de configuration de l'application.

Ce module centralise toutes les variables de configuration de l'application.
Au lieu de disperser les configurations dans tout le code, on les regroupe ici.

Pourquoi c'est important :
- Facilite la maintenance : un seul endroit pour changer la configuration
- Permet de surcharger avec des variables d'environnement
- Évite les valeurs codées en dur dans le code
"""

import os
from pathlib import Path

class Settings:
    """
    Classe qui contient toutes les configurations de l'application.
    
    Comment ça marche :
    - os.getenv("NOM_VARIABLE", "valeur_par_defaut") : 
      * Cherche d'abord une variable d'environnement nommée "NOM_VARIABLE"
      * Si elle n'existe pas, utilise "valeur_par_defaut"
      * Les variables d'environnement permettent de configurer sans modifier le code
    
    Variables d'environnement :
    - MODEL_VERSION : Version du modèle (ex: "1.0.0", "1.1.0")
    - MODEL_PATH : Chemin vers le fichier du modèle (ex: "models/model.joblib")
    """
    
    # Version du modèle (utilisée pour la traçabilité)
    # Exemple : si vous entraînez un nouveau modèle, changez la version
    # Cela permet de savoir quel modèle a fait quelle prédiction
    MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
    
    # Chemin vers le fichier du modèle
    # Par défaut : "models/logistic_regression_model.joblib"
    # Mais dans ce projet, on utilise XGBoost, donc ce chemin est juste un exemple
    # Le vrai modèle est dans models/xgb_booster.json
    MODEL_PATH = os.getenv("MODEL_PATH", "models/logistic_regression_model.joblib")
    
    # Dossier contenant tous les modèles et artefacts
    # Path("models") crée un objet Path (plus pratique que les strings pour manipuler les chemins)
    # Exemple : MODELS_DIR / "model.joblib" = "models/model.joblib"
    MODELS_DIR = Path("models")

# Créer une instance unique de Settings
# Cette instance sera importée partout dans l'application
# C'est un pattern "singleton" : une seule instance partagée
settings = Settings()

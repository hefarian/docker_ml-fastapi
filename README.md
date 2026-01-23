# API de Prédiction d'Attrition - Machine Learning

API FastAPI pour prédire l'attrition des employés à l'aide d'un modèle XGBoost optimisé.

**NB : Toute la documentation et les commentaires du code ont été corrigés et mis en forme par une IA générative, en « mode tutoriel », afin d’en améliorer la compréhension.**

## 📋 Table des matières

- [Description](#description)
- [Architecture](#architecture)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Tests](#tests)
- [Entraînement du modèle](#entraînement-du-modèle)
- [Déploiement](#déploiement)
- [CI/CD](#cicd)

## 🎯 Description

Ce projet déploie un modèle de machine learning (XGBoost) pour prédire l'attrition des employés. Le modèle a été entraîné sur des données d'attrition d'entreprise et utilise PostgreSQL pour stocker les données et les prédictions.

### Fonctionnalités

- ✅ API REST avec FastAPI
- ✅ Modèle XGBoost optimisé avec Optuna
- ✅ Base de données PostgreSQL
- ✅ Enregistrement automatique des prédictions
- ✅ Documentation automatique (Swagger/OpenAPI)
- ✅ Tests unitaires avec pytest
- ✅ Docker et Docker Compose
- ✅ Pipeline CI/CD complet

## 🏗️ Architecture

```
docker_ml-fastapi/
├── .github/workflows/
│   ├── ci.yml                  # Github Action : Intégration continue
│   └── deploy-render.yml       # Github Action : Déploiement continu sur render.com
├── app/
│   ├── main.py                 # API FastAPI principale
│   ├── model.py                # Chargement et utilisation du modèle
│   ├── schemas.py              # Schémas Pydantic
│   ├── settings.py             # Configuration
│   ├── database.py             # Gestion PostgreSQL
│   ├── train_model.py          # Script d'entraînement
│   └── requirements.txt        # Dépendances Python
├── db/
│   ├── db.py                   # Utilitaires SQLAlchemy
│   └── init/
│       ├── attrition.sql       # Script SQL avec données
│       └── prediction.sql      # Script SQL de création de la table predictions
├── tests/
│   └── test_api.py             # Tests unitaires
├── models/                     # Modèles entraînés 
│   ├── feature_names.joblib    # Une liste Python des noms de colonnes exactement dans l’ordre que le modèle a vu pendant l’entraînement
│   ├── one_hot_encoder.joblib  # L’objet OneHotEncoder (scikit‑learn) déjà fit : catégories apprises, gestion des valeurs inconnues, noms des colonnes générées.
│   ├── ordinal_encoder.joblib  # L’objet OrdinalEncoder (scikit‑learn) déjà fit : mapping catégorie → entier (et la stratégie pour valeurs inconnues).
│   ├── xgb_best_params.joblib  # les meilleurs hyperparamètres trouvés par Optuna
│   └── xgb_booster.json        # Le booster XGBoost entraîné (structure des arbres, poids des feuilles, paramètres internes) sérialisé en JSON.
├── Dockerfile
├── docker-compose.yml
├── GUIDE_TESTS.md
├── pyproject.toml              # Fichier de configuration de pytest pour le projet en Python
├── pytest.ini                  # Fichier de configuration de pytest pour le projet sous Windows
└── README.md
```

## 🚀 Installation

### Prérequis

- Docker et Docker Compose
- Python 3.11+ (pour développement local)

### Installation avec Docker (Recommandé)

```bash
# Cloner le dépôt
git clone https://github.com/hefarian/docker_ml-fastapi
cd docker_ml-fastapi

# Démarrer les services
docker-compose up --build
```

L'API sera accessible sur `http://localhost:8090`

### Installation locale

```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r app/requirements.txt
```

## 📖 Utilisation

### Documentation interactive

Accédez à la documentation Swagger :
```
http://localhost:8090/docs
```

### Exemple de prédiction

```bash
curl -X POST "http://localhost:8090/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "employee_data": {
      "age": 35,
      "genre": "M",
      "revenu_mensuel": 5000,
      ...
    }
  }'
```

## 🧪 Tests

Voir [GUIDE_TESTS.md](GUIDE_TESTS.md) pour le guide complet.

### Exécution rapide

```bash
# Tests basiques
pytest tests/ -v

# Avec couverture
pytest tests/ --cov=app --cov-report=html
```

## 🎯 Entraînement du modèle

```bash
# Localement
export DATABASE_URL=postgresql://postgres:password@localhost:5432/mydatabase
python app/train_model.py

# Avec Docker
docker-compose run --rm \
  -e DATABASE_URL=postgresql://postgres:password@db:5432/mydatabase \
  -v "%CD%\models:/app/models" \
  api python train_model.py
```


### Déploiement sur Render.com

1. Connecter votre dépôt Git à Render
2. Configurer les variables d'environnement
3. Le déploiement se fait automatiquement via GitHub Actions

## 🔄 CI/CD

### Workflows GitHub Actions

1. **CI - Tests et Validation** (`.github/workflows/ci.yml`)
   - Exécute les tests à chaque push/PR
   - Vérifie le linting et le formatage
   - Génère un rapport de couverture

2. **Deploy to Render** (`.github/workflows/deploy-render.yml`)
   - Déploiement automatique après tests réussis
   - Déploiement en dev (branche `dev`) ou prod (branche `main`)

### Configuration des secrets GitHub

Dans les paramètres du dépôt, ajouter :
- `RENDER_HOOK_DEV` : Webhook Render pour dev
- `RENDER_HOOK_PROD` : Webhook Render pour prod
- `RENDER_URL_DEV` : URL de l'API dev (optionnel)
- `RENDER_URL_PROD` : URL de l'API prod (optionnel)

## 📚 Documentation complète

- [GUIDE_TESTS.md](GUIDE_TESTS.md) - Guide d'utilisation des tests
## 📄 Licence

Ce projet est un projet de formation OpenClassrooms.

## 👤 Auteur

Hefarian

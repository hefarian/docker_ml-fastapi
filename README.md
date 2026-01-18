# API de Prédiction d'Attrition - Machine Learning

API FastAPI pour prédire l'attrition des employés à l'aide d'un modèle XGBoost optimisé.

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
├── app/
│   ├── main.py              # API FastAPI principale
│   ├── model.py             # Chargement et utilisation du modèle
│   ├── schemas.py           # Schémas Pydantic
│   ├── settings.py          # Configuration
│   ├── database.py          # Gestion PostgreSQL
│   ├── train_model.py       # Script d'entraînement
│   └── requirements.txt     # Dépendances Python
├── db/
│   ├── db.py                # Utilitaires SQLAlchemy
│   └── init/
│       └── attrition.sql    # Script SQL avec données
├── tests/
│   └── test_api.py          # Tests unitaires
├── models/                  # Modèles entraînés (généré)
├── .github/workflows/       # CI/CD
│   ├── ci.yml              # Tests automatiques
│   ├── train_model.yml      # Entraînement automatique
│   └── deploy-render.yml    # Déploiement Render
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 🚀 Installation

### Prérequis

- Docker et Docker Compose
- Python 3.11+ (pour développement local)

### Installation avec Docker (Recommandé)

```bash
# Cloner le dépôt
git clone <votre-repo>
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

Voir [GUIDE_TRAIN_DEPLOY.md](GUIDE_TRAIN_DEPLOY.md) pour le guide complet.

### Exécution rapide

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

## 🚀 Déploiement

Voir [GUIDE_TRAIN_DEPLOY.md](GUIDE_TRAIN_DEPLOY.md) pour le guide complet.

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

2. **Train Model** (`.github/workflows/train_model.yml`)
   - Entraînement manuel ou automatique (hebdomadaire)
   - Sauvegarde les modèles en artifacts

3. **Deploy to Render** (`.github/workflows/deploy-render.yml`)
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
- [GUIDE_TRAIN_DEPLOY.md](GUIDE_TRAIN_DEPLOY.md) - Guide d'entraînement et déploiement

## 📄 Licence

Ce projet est un projet de formation OpenClassrooms.

## 👤 Auteur

Hefarian

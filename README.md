# API de Prédiction d'Attrition - Machine Learning

API FastAPI pour prédire l'attrition des employés à l'aide d'un modèle de régression logistique optimisé avec Elastic Net.

## 📋 Table des matières

- [Description](#description)
- [Architecture](#architecture)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [API Documentation](#api-documentation)
- [Tests](#tests)
- [Déploiement](#déploiement)

## 🎯 Description

Ce projet déploie un modèle de machine learning (régression logistique avec régularisation Elastic Net) pour prédire l'attrition des employés. Le modèle a été entraîné sur des données d'attrition d'entreprise et utilise PostgreSQL pour stocker les données et les prédictions.

### Fonctionnalités

- ✅ API REST avec FastAPI
- ✅ Modèle de régression logistique avec Elastic Net
- ✅ Base de données PostgreSQL
- ✅ Enregistrement automatique des prédictions
- ✅ Documentation automatique (Swagger/OpenAPI)
- ✅ Tests unitaires avec pytest
- ✅ Docker et Docker Compose
- ✅ Pipeline CI/CD prêt

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
│   ├── load_data.py         # Chargement des données CSV → PostgreSQL
│   └── requirements.txt     # Dépendances Python
├── db/
│   ├── db.py                # Utilitaires SQLAlchemy
│   └── init/
│       └── 01_attrition.sql # Script SQL d'initialisation
├── tests/
│   └── test_api.py          # Tests unitaires
├── models/                  # Modèles entraînés (généré)
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 🚀 Installation

### Prérequis

- Docker et Docker Compose
- Python 3.11+ (pour développement local)
- PostgreSQL 15+ (ou via Docker)

### Installation avec Docker (Recommandé)

1. **Cloner le dépôt**
```bash
git clone <votre-repo>
cd docker_ml-fastapi
```

2. **Créer un fichier `.env`** (optionnel)
```bash
DATABASE_URL=postgresql://postgres:password@db:5432/mydatabase
MODEL_VERSION=1.0.0
```

3. **Construire et démarrer les conteneurs**
```bash
docker-compose up --build
```

L'API sera accessible sur `http://localhost:8090`

### Installation locale (Développement)

1. **Créer un environnement virtuel**
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```

2. **Installer les dépendances**
```bash
pip install -r app/requirements.txt
```

3. **Configurer PostgreSQL**
   - Créer une base de données `mydatabase`
   - Exécuter le script `db/init/01_attrition.sql`

4. **Charger les données** (si vous avez les CSV)
```bash
export DATABASE_URL="postgresql://postgres:password@localhost:5432/mydatabase"
python app/load_data.py
```

5. **Entraîner le modèle**
```bash
python app/train_model.py
```

6. **Démarrer l'API**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8090
```

## 📖 Utilisation

### 1. Charger les données dans PostgreSQL

Si vous avez les fichiers CSV (`extrait_sirh.csv`, `extrait_eval.csv`, `extrait_sondage.csv`), placez-les dans un dossier `data/` et exécutez :

```bash
python app/load_data.py
```

### 2. Entraîner le modèle

```bash
python app/train_model.py
```

Le script va :
- Charger les données depuis PostgreSQL
- Préparer les features (encodage, variables calculées)
- Entraîner un modèle de régression logistique avec Elastic Net
- Optimiser les hyperparamètres avec GridSearchCV
- Sauvegarder le modèle dans `models/`

### 3. Utiliser l'API

#### Documentation interactive

Accédez à la documentation Swagger :
```
http://localhost:8090/docs
```

#### Exemple de prédiction

```bash
curl -X POST "http://localhost:8090/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "employee_data": {
      "age": 35,
      "genre": "M",
      "revenu_mensuel": 5000,
      "statut_marital": "Marié(e)",
      "departement": "R&D",
      "poste": "Data Scientist",
      "nombre_experiences_precedentes": 3,
      "annee_experience_totale": 8,
      "annees_dans_l_entreprise": 5,
      "annees_dans_le_poste_actuel": 2,
      "satisfaction_employee_environnement": 4,
      "note_evaluation_precedente": 4.0,
      "niveau_hierarchique_poste": 2,
      "satisfaction_employee_nature_travail": 4,
      "satisfaction_employee_equipe": 4,
      "satisfaction_employee_equilibre_pro_perso": 3,
      "note_evaluation_actuelle": 4.5,
      "heure_supplementaires": "Non",
      "nombre_participation_pee": 2,
      "nb_formations_suivies": 3,
      "distance_domicile_travail": 10,
      "niveau_education": 3,
      "domaine_etude": "Data Science",
      "frequence_deplacement": "Occasionnel",
      "annees_depuis_la_derniere_promotion": 2,
      "annes_sous_responsable_actuel": 1,
      "augmentation_salaire_precedente": 0.05
    }
  }'
```

## 📚 API Documentation

### Endpoints

#### `GET /health`
Vérifie l'état de l'API, du modèle et de la base de données.

**Réponse:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "db_connected": true
}
```

#### `POST /predict`
Prédit l'attrition pour un employé.

**Body:**
```json
{
  "employee_data": {
    "age": 35,
    "genre": "M",
    ...
  }
}
```

**Réponse:**
```json
{
  "prediction": 0,
  "probability": 0.2345,
  "model_version": "1.0.0",
  "prediction_id": 123
}
```

#### `GET /predictions`
Récupère les dernières prédictions enregistrées.

**Paramètres:**
- `limit` (optionnel): Nombre de prédictions à retourner (défaut: 100)

#### `GET /db-test`
Teste la connexion à la base de données.

## 🧪 Tests

### Exécuter les tests

```bash
pytest tests/ -v
```

### Avec couverture de code

```bash
pytest tests/ --cov=app --cov-report=html
```

Les rapports de couverture seront générés dans `htmlcov/`.

## 🐳 Déploiement

### Docker Compose

Le fichier `docker-compose.yml` configure :
- **API FastAPI** sur le port 8090
- **PostgreSQL** sur le port 5432

### Variables d'environnement

- `DATABASE_URL`: URL de connexion PostgreSQL
- `MODEL_VERSION`: Version du modèle (défaut: 1.0.0)

### Déploiement sur Render.com

1. Connecter votre dépôt Git à Render
2. Configurer les variables d'environnement
3. Déployer avec le Dockerfile fourni

## 🔧 Modèle de Machine Learning

### Caractéristiques

- **Algorithme**: Régression logistique avec régularisation Elastic Net
- **Optimisation**: GridSearchCV avec validation croisée (5 folds)
- **Métriques**: ROC-AUC, AUC-PR, F1-score
- **Features**: 39 variables après encodage et création de variables calculées

### Variables calculées

- `ratio_anciennete`: Ancienneté / Expérience totale
- `ratio_poste`: Années dans le poste / Années dans l'entreprise
- `ecart_evaluation`: Différence entre évaluations
- `ratio_salaire_niveau`: Salaire / Niveau hiérarchique
- `ratio_formations`: Formations / Ancienneté
- `indice_recente_promo`: Indice de récence de promotion

## 📝 Structure de la base de données

### Tables

- **sirh**: Données RH des employés
- **performance**: Évaluations de performance
- **sondage**: Données de sondage bien-être
- **predictions**: Enregistrement des prédictions (inputs/outputs)

## 🤝 Contribution

1. Créer une branche (`git checkout -b feature/amélioration`)
2. Commit les changements (`git commit -m 'Ajout d'une fonctionnalité'`)
3. Push vers la branche (`git push origin feature/amélioration`)
4. Ouvrir une Pull Request

## 📄 Licence

Ce projet est un projet de formation OpenClassrooms.

## 👤 Auteur

Gregory Crespin - Projet 4 - Déploiement d'un modèle ML

---

**Note**: Ce projet nécessite que le modèle soit entraîné avant de pouvoir faire des prédictions. Exécutez `python app/train_model.py` après avoir chargé les données dans PostgreSQL.

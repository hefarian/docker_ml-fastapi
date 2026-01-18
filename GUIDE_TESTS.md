# Guide d'utilisation des tests

## 📋 Exécution des tests

### Prérequis

```bash
# Installer les dépendances de test
pip install -r app/requirements.txt
```

### Exécution basique

```bash
# Depuis la racine du projet
pytest tests/ -v
```

### Exécution avec couverture de code

```bash
# Générer un rapport de couverture HTML
pytest tests/ --cov=app --cov-report=html

# Ouvrir le rapport
# Windows: start htmlcov/index.html
# Linux/Mac: open htmlcov/index.html
```

### Exécution d'un test spécifique

```bash
# Un seul test
pytest tests/test_api.py::test_health_ok -v

# Tous les tests d'un fichier
pytest tests/test_api.py -v
```

### Exécution avec Docker

```bash
# Exécuter les tests dans le conteneur
docker-compose run --rm api pytest tests/ -v

# Avec couverture
docker-compose run --rm api pytest tests/ --cov=app --cov-report=term
```

## 🧪 Tests disponibles

### Tests d'endpoints

- `test_health_ok` : Vérifie que l'endpoint `/health` fonctionne
- `test_root_ok` : Vérifie que l'endpoint racine fonctionne
- `test_predict_with_valid_data` : Test de prédiction avec données valides
- `test_get_predictions` : Test de récupération des prédictions
- `test_db_test` : Test de connexion à la base de données

### Tests de validation

- `test_predict_validation_error_missing_field` : Test avec champ manquant
- `test_predict_validation_error_invalid_value` : Test avec valeur invalide
- `test_predict_validation_error_wrong_type` : Test avec type incorrect
- `test_schema_employee_data` : Validation du schéma Pydantic
- `test_schema_predict_request` : Validation de la requête

## 📊 Interprétation des résultats

### Codes de statut acceptés

Certains tests acceptent plusieurs codes de statut car ils peuvent échouer si :
- Le modèle n'est pas encore entraîné (503 au lieu de 200)
- La base de données n'est pas connectée (500 au lieu de 200)

### Exemple de sortie

```
tests/test_api.py::test_health_ok PASSED
tests/test_api.py::test_predict_with_valid_data PASSED
tests/test_api.py::test_predict_validation_error_missing_field PASSED
...
```

## 🔧 Configuration pour les tests

### Variables d'environnement

Les tests utilisent les variables d'environnement par défaut. Pour les modifier :

```bash
export DATABASE_URL=postgresql://postgres:password@localhost:5432/mydatabase
pytest tests/ -v
```

### Mode verbose

```bash
# Afficher plus de détails
pytest tests/ -v -s

# Afficher les print statements
pytest tests/ -v -s --capture=no
```

## 🚨 Résolution de problèmes

### Erreur : ModuleNotFoundError

```bash
# Installer les dépendances
pip install -r app/requirements.txt
```

### Erreur : Connexion à la base de données

Les tests peuvent fonctionner sans base de données (certains tests acceptent 500).
Pour tester avec la base :

```bash
# Démarrer PostgreSQL
docker-compose up -d db

# Attendre qu'il soit prêt
timeout /t 10

# Exécuter les tests
pytest tests/ -v
```

### Erreur : Modèle non trouvé

Certains tests acceptent 503 si le modèle n'est pas chargé.
Pour tester avec le modèle :

```bash
# Entraîner le modèle d'abord
python app/train_model.py

# Puis exécuter les tests
pytest tests/ -v
```

## 📈 Rapport de couverture

### Générer un rapport HTML

```bash
pytest tests/ --cov=app --cov-report=html
```

Le rapport sera dans `htmlcov/index.html`.

### Objectif de couverture

- **Minimum recommandé** : 70%
- **Objectif** : 80%+
- **Actuel** : Vérifier avec `pytest tests/ --cov=app --cov-report=term`

## 🔄 Intégration dans CI/CD

Les tests sont automatiquement exécutés dans GitHub Actions lors des push.

Voir `.github/workflows/` pour les workflows de CI/CD.

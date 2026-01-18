# Guide d'entraînement et de déploiement

## 🎯 Entraînement du modèle

### Prérequis

1. PostgreSQL doit être démarré avec les données chargées
2. Les dépendances Python installées

### Méthode 1 : Localement

```bash
# 1. Démarrer PostgreSQL
docker-compose up -d db

# 2. Attendre que PostgreSQL soit prêt
timeout /t 10  # Windows
# ou
sleep 10       # Linux/Mac

# 3. Définir l'URL de la base de données
set DATABASE_URL=postgresql://postgres:password@localhost:5432/mydatabase  # Windows
# ou
export DATABASE_URL=postgresql://postgres:password@localhost:5432/mydatabase  # Linux/Mac

# 4. Entraîner le modèle
python app/train_model.py
```

### Méthode 2 : Avec Docker

```bash
# 1. Démarrer PostgreSQL
docker-compose up -d db

# 2. Attendre que PostgreSQL soit prêt
timeout /t 10

# 3. Entraîner dans Docker
docker-compose run --rm \
  -e DATABASE_URL=postgresql://postgres:password@db:5432/mydatabase \
  -v "%CD%\models:/app/models" \
  api python train_model.py
```

### Variables d'environnement pour l'entraînement

```bash
# Nombre d'essais Optuna (défaut: 60)
export OPTUNA_TRIALS=60

# Timeout en secondes (défaut: None)
export OPTUNA_TIMEOUT=1800

# URL de la base de données
export DATABASE_URL=postgresql://postgres:password@localhost:5432/mydatabase
```

### Résultat de l'entraînement

Le modèle sera sauvegardé dans `models/` :
- `xgb_booster.json` : Modèle XGBoost
- `onehot_encoder.joblib` : Encodeur OneHot
- `ordinal_encoder.joblib` : Encodeur ordinal
- `feature_names.joblib` : Liste des features
- `xgb_best_params.joblib` : Meilleurs hyperparamètres

## 🚀 Déploiement

### Déploiement sur Render.com

#### 1. Configuration Render

1. Créer un nouveau service Web sur Render
2. Connecter votre dépôt GitHub
3. Configurer les variables d'environnement :
   ```
   DATABASE_URL=<votre_url_postgresql>
   MODEL_VERSION=1.0.0
   ```

#### 2. Build Command

```bash
pip install -r app/requirements.txt
```

#### 3. Start Command

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

#### 4. Health Check Path

```
/health
```

### Déploiement avec Docker

#### Build et push de l'image

```bash
# Build l'image
docker build -t votre-registry/api-attrition:latest .

# Push vers le registry
docker push votre-registry/api-attrition:latest
```

#### Déploiement sur un serveur

```bash
# Pull l'image
docker pull votre-registry/api-attrition:latest

# Run le conteneur
docker run -d \
  -p 8090:8090 \
  -e DATABASE_URL=postgresql://user:pass@host:5432/db \
  -e MODEL_VERSION=1.0.0 \
  votre-registry/api-attrition:latest
```

## 🔄 Workflow CI/CD

### GitHub Actions - Entraînement

Le workflow `.github/workflows/train_model.yml` permet d'entraîner le modèle manuellement :

1. Aller dans l'onglet "Actions" de GitHub
2. Sélectionner "Train model"
3. Cliquer sur "Run workflow"

### GitHub Actions - Déploiement

Le workflow `.github/workflows/deploy-render.yml` déploie automatiquement sur Render :

- **Branche `main`** → Déploiement en production
- **Branche `dev`** → Déploiement en développement

#### Configuration des secrets GitHub

Dans les paramètres du dépôt GitHub, ajouter :

- `RENDER_HOOK_DEV` : URL du webhook Render pour dev
- `RENDER_HOOK_PROD` : URL du webhook Render pour prod

### Pipeline complet

1. **Push sur `dev` ou `main`**
   ↓
2. **Tests automatiques** (pytest)
   ↓
3. **Déploiement automatique** sur Render
   ↓
4. **Health check** automatique

## 📋 Checklist de déploiement

### Avant le déploiement

- [ ] Modèle entraîné et sauvegardé dans `models/`
- [ ] Tests passent : `pytest tests/ -v`
- [ ] Variables d'environnement configurées
- [ ] Base de données accessible depuis Render
- [ ] Secrets GitHub configurés (pour CI/CD)

### Après le déploiement

- [ ] Vérifier `/health` retourne `200`
- [ ] Vérifier `/docs` accessible (Swagger)
- [ ] Tester une prédiction via `/predict`
- [ ] Vérifier les logs pour erreurs

## 🔍 Vérification post-déploiement

### Test de santé

```bash
curl https://votre-api.render.com/health
```

Réponse attendue :
```json
{
  "status": "ok",
  "model_loaded": true,
  "db_connected": true
}
```

### Test de prédiction

```bash
curl -X POST https://votre-api.render.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "employee_data": {
      "age": 35,
      "genre": "M",
      ...
    }
  }'
```

## 🐛 Résolution de problèmes

### Le modèle n'est pas chargé

1. Vérifier que les fichiers sont dans `models/`
2. Vérifier les permissions
3. Vérifier le chemin dans `app/model.py`

### Erreur de connexion à la base

1. Vérifier `DATABASE_URL`
2. Vérifier que PostgreSQL est accessible depuis Render
3. Vérifier les credentials

### Erreur lors du déploiement

1. Vérifier les logs Render
2. Vérifier que toutes les dépendances sont dans `requirements.txt`
3. Vérifier que le Dockerfile est correct

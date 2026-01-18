# Guide : Exécuter train_model.py sur Render.com

## 📋 Vue d'ensemble

Après le déploiement de votre API sur Render.com, vous avez plusieurs options pour exécuter le script d'entraînement `train_model.py`. Ce guide explique chaque méthode avec des exemples pratiques.

## 🎯 Méthode 1 : Via le Shell Render (Recommandé pour tests)

### Étape 1 : Accéder au Shell Render

1. Connectez-vous à votre compte Render.com
2. Allez dans votre service (API)
3. Cliquez sur l'onglet **"Shell"** dans le menu latéral
4. Un terminal s'ouvre directement dans votre conteneur

### Étape 2 : Exécuter le script

```bash
# Vérifier que vous êtes dans le bon répertoire
pwd
# Devrait afficher : /app

# Vérifier que les fichiers sont présents
ls -la train_model.py

# Vérifier que DATABASE_URL est définie
echo $DATABASE_URL

# Exécuter le script d'entraînement
python train_model.py
```

### Avantages
- ✅ Simple et direct
- ✅ Voir les logs en temps réel
- ✅ Utile pour tester et déboguer

### Inconvénients
- ❌ Nécessite une connexion manuelle
- ❌ Le shell se ferme après déconnexion
- ❌ Pas automatisé

---

## 🎯 Méthode 2 : Créer un endpoint API dédié (Recommandé pour production)

### Étape 1 : Ajouter un endpoint dans `app/main.py`

Ajoutez cet endpoint à votre fichier `app/main.py` (après les autres endpoints) :

```python
@app.post("/admin/train-model")
async def trigger_training():
    """
    Endpoint pour déclencher l'entraînement du modèle.
    
    ⚠️ SÉCURITÉ : Ajouter une authentification en production !
    
    COMMENT ÇA MARCHE ?
    ===================
    Cet endpoint lance le script train_model.py en arrière-plan.
    Le modèle sera entraîné et sauvegardé dans le dossier models/.
    
    NOTE IMPORTANTE :
    =================
    - L'entraînement peut prendre plusieurs minutes
    - L'API peut être temporairement indisponible pendant l'entraînement
    - En production, utilisez une queue (Celery, RQ) pour éviter les timeouts
    """
    import subprocess
    import os
    import asyncio
    
    # Vérifier que DATABASE_URL est définie
    if not os.getenv("DATABASE_URL"):
        raise HTTPException(
            status_code=500,
            detail="DATABASE_URL non définie"
        )
    
    try:
        # Exécuter le script d'entraînement en arrière-plan
        # subprocess.Popen : lance un processus sans attendre
        process = subprocess.Popen(
            ["python", "train_model.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="/app"  # Répertoire de travail
        )
        
        return {
            "status": "started",
            "message": "Entraînement démarré",
            "pid": process.pid,
            "note": "L'entraînement s'exécute en arrière-plan. Vérifiez les logs Render pour suivre la progression."
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du démarrage de l'entraînement: {e}"
        )
```

### Étape 2 : Appeler l'endpoint

```bash
# Depuis votre machine locale ou un autre service
curl -X POST https://votre-api.render.com/admin/train-model
```

### ⚠️ Amélioration : Ajouter une authentification

Pour sécuriser l'endpoint, ajoutez une vérification de token :

```python
from fastapi import Header, HTTPException
import os

# Token d'administration (à définir dans les variables d'environnement Render)
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "changez-moi-en-production")

@app.post("/admin/train-model")
async def trigger_training(authorization: str = Header(None)):
    """
    Endpoint sécurisé pour déclencher l'entraînement.
    
    AUTHENTIFICATION :
    ==================
    Envoyez le token dans le header Authorization :
    Authorization: Bearer votre-token-secret
    """
    
    # Vérifier le token d'authentification
    if not authorization or authorization != f"Bearer {ADMIN_TOKEN}":
        raise HTTPException(
            status_code=401,
            detail="Token d'authentification invalide ou manquant"
        )
    
    # ... reste du code de l'endpoint ...
```

### Utilisation avec authentification

```bash
# Définir le token (remplacez par votre token réel)
TOKEN="votre-token-secret"

# Appeler l'endpoint avec authentification
curl -X POST https://votre-api.render.com/admin/train-model \
  -H "Authorization: Bearer $TOKEN"
```

### Configurer le token dans Render

Dans les paramètres de votre service Render, ajoutez la variable d'environnement :
```
ADMIN_TOKEN=votre-token-secret-tres-long-et-complexe
```

### Avantages
- ✅ Automatisable (peut être appelé depuis un autre service)
- ✅ Intégrable dans un workflow CI/CD
- ✅ Peut être déclenché à distance

### Inconvénients
- ❌ Nécessite une authentification (sécurité)
- ❌ L'entraînement peut être long (risque de timeout)
- ❌ Pas de suivi de progression en temps réel

---

## 🎯 Méthode 3 : Via GitHub Actions (Recommandé pour automatisation)

### Étape 1 : Créer un workflow GitHub Actions

Créez un fichier `.github/workflows/train_on_render.yml` :

```yaml
name: Train Model on Render

on:
  workflow_dispatch:  # Lancement manuel depuis GitHub
  schedule:
    # Tous les dimanches à 2h du matin (UTC)
    - cron: '0 2 * * 0'

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - name: Trigger Training via API
        env:
          RENDER_URL: ${{ secrets.RENDER_URL_PROD }}
          ADMIN_TOKEN: ${{ secrets.ADMIN_TOKEN }}
        run: |
          echo "🚀 Déclenchement de l'entraînement sur Render..."
          
          # Appeler l'endpoint API pour démarrer l'entraînement
          response=$(curl -X POST "$RENDER_URL/admin/train-model" \
            -H "Authorization: Bearer $ADMIN_TOKEN" \
            -w "\n%{http_code}")
          
          http_code=$(echo "$response" | tail -n1)
          body=$(echo "$response" | head -n-1)
          
          if [ "$http_code" -eq 200 ]; then
            echo "✅ Entraînement démarré avec succès"
            echo "$body"
          else
            echo "❌ Erreur lors du démarrage de l'entraînement"
            echo "Code HTTP: $http_code"
            echo "Réponse: $body"
            exit 1
          fi
```

### Étape 2 : Configurer les secrets GitHub

Dans les paramètres de votre dépôt GitHub (Settings > Secrets and variables > Actions), ajoutez :

- `RENDER_URL_PROD` : URL de votre API Render (ex: `https://votre-api.onrender.com`)
- `ADMIN_TOKEN` : Token d'authentification (même valeur que dans Render)

### Étape 3 : Lancer le workflow

1. Allez dans l'onglet **"Actions"** de GitHub
2. Sélectionnez **"Train Model on Render"**
3. Cliquez sur **"Run workflow"**
4. Sélectionnez la branche et cliquez sur **"Run workflow"**

### Avantages
- ✅ Entièrement automatisé
- ✅ Peut être planifié (hebdomadaire, mensuel)
- ✅ Traçabilité dans GitHub Actions
- ✅ Pas besoin de se connecter à Render

### Inconvénients
- ❌ Nécessite une configuration initiale
- ❌ Nécessite l'endpoint API `/admin/train-model`

---

## 🎯 Méthode 4 : Via un service séparé Render (Recommandé pour production)

### Étape 1 : Créer un nouveau service Render

1. Dans Render Dashboard, cliquez sur **"New +"**
2. Sélectionnez **"Background Worker"**
3. Configurez :
   - **Name** : `train-model-worker`
   - **Environment** : Python 3
   - **Build Command** : `pip install -r app/requirements.txt`
   - **Start Command** : `python app/train_model.py`
   - **Plan** : Free ou Starter (selon vos besoins)

### Étape 2 : Configurer les variables d'environnement

Dans les paramètres du service, ajoutez les mêmes variables que votre API :
- `DATABASE_URL` : URL de votre base PostgreSQL Render
- `MODEL_VERSION` : Version du modèle (ex: `1.0.0`)
- `OPTUNA_TRIALS` : 60 (ou autre valeur)
- `OPTUNA_TIMEOUT` : 1800 (30 minutes)

### Étape 3 : Configurer le service

**Option A : Service manuel (recommandé)**
- Désactivez le service par défaut
- Activez-le manuellement quand vous voulez entraîner le modèle
- Le service s'arrête automatiquement après l'entraînement

**Option B : Service automatique**
- Le service s'exécute en boucle
- Modifiez `train_model.py` pour ajouter une boucle avec `time.sleep()`

### Avantages
- ✅ Isolation complète (n'affecte pas l'API)
- ✅ Peut tourner longtemps sans timeout
- ✅ Logs dédiés
- ✅ Peut être activé/désactivé facilement

### Inconvénients
- ❌ Nécessite un service séparé (coût supplémentaire si payant)
- ❌ Configuration supplémentaire

---

## 🎯 Méthode 5 : Via Render Cron Jobs (Fonctionnalité payante)

Si vous avez un plan payant Render, vous pouvez utiliser les Cron Jobs :

1. Allez dans votre service
2. Cliquez sur **"Cron Jobs"** dans le menu
3. Créez un nouveau job :
   - **Schedule** : `0 2 * * 0` (tous les dimanches à 2h)
   - **Command** : `python app/train_model.py`
   - **Timezone** : UTC (ou votre fuseau horaire)

---

## 📊 Comparaison des méthodes

| Méthode | Automatisation | Coût | Complexité | Recommandation |
|---------|----------------|------|------------|----------------|
| Shell Render | ❌ | Gratuit | ⭐ Facile | Tests/Debug |
| Endpoint API | ⚠️ Partiel | Gratuit | ⭐⭐ Moyen | Développement |
| GitHub Actions | ✅ | Gratuit | ⭐⭐⭐ Avancé | Production |
| Service séparé | ⚠️ Manuel | Payant* | ⭐⭐ Moyen | Production |
| Cron Jobs | ✅ | Payant | ⭐ Facile | Production |

*Gratuit si vous utilisez le plan Free de Render

---

## 🔧 Configuration recommandée pour Render

### Variables d'environnement à définir dans Render

Dans les paramètres de votre service Render, ajoutez :

```
DATABASE_URL=postgresql://user:password@host:5432/database
MODEL_VERSION=1.0.0
OPTUNA_TRIALS=60
OPTUNA_TIMEOUT=1800
ADMIN_TOKEN=votre-token-secret  # Pour l'endpoint /admin/train-model
```

### Structure des fichiers sur Render

Assurez-vous que votre configuration Render inclut :

**Build Command** :
```bash
pip install -r app/requirements.txt
```

**Start Command** :
```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

**Root Directory** :
```
app
```

### Vérifier que les fichiers sont présents

Dans le Shell Render :

```bash
# Vérifier la structure
ls -la
# Devrait afficher : train_model.py, main.py, etc.

# Vérifier les permissions
chmod +x train_model.py

# Vérifier PYTHONPATH
echo $PYTHONPATH
# Devrait être : /app (ou vide, ce qui est OK)
```

---

## 🚨 Problèmes courants et solutions

### Problème 1 : "Module not found"

**Symptôme** :
```
ModuleNotFoundError: No module named 'app'
```

**Solution** :
```bash
# Dans le Shell Render, vérifier PYTHONPATH
echo $PYTHONPATH

# Si vide, définir :
export PYTHONPATH=/app

# Ou modifier le script pour ajouter au début :
import sys
sys.path.insert(0, '/app')
```

### Problème 2 : "DATABASE_URL not found"

**Symptôme** :
```
RuntimeError: DATABASE_URL non défini
```

**Solution** :
1. Vérifier que la variable est définie dans Render Dashboard
2. Vérifier l'URL (doit pointer vers votre base PostgreSQL Render)
3. Format attendu : `postgresql://user:password@host:5432/database`

### Problème 3 : "Timeout" ou "Request timeout"

**Symptôme** :
L'entraînement prend trop de temps et l'API timeout.

**Solution** :
- Utiliser un service séparé (Background Worker)
- Réduire `OPTUNA_TRIALS` (ex: 30 au lieu de 60)
- Augmenter `OPTUNA_TIMEOUT` si nécessaire
- Utiliser l'endpoint API qui lance en arrière-plan

### Problème 4 : "Permission denied"

**Symptôme** :
```
PermissionError: [Errno 13] Permission denied
```

**Solution** :
```bash
# Dans le Shell Render
chmod +x train_model.py
chmod +r train_model.py
```

### Problème 5 : "Cannot write to models/"

**Symptôme** :
```
PermissionError: cannot write to models/
```

**Solution** :
```bash
# Créer le dossier models/ avec les bonnes permissions
mkdir -p models
chmod 755 models
```

---

## 📝 Exemple complet : Script d'entraînement automatisé

Créez un fichier `scripts/train_on_render.sh` :

```bash
#!/bin/bash
# Script pour entraîner le modèle sur Render

set -e  # Arrêter en cas d'erreur

echo "🚀 Démarrage de l'entraînement sur Render..."
echo "Date: $(date)"

# Vérifier les variables d'environnement
if [ -z "$DATABASE_URL" ]; then
    echo "❌ ERREUR: DATABASE_URL non définie"
    exit 1
fi

echo "✅ DATABASE_URL définie"
echo "✅ OPTUNA_TRIALS: ${OPTUNA_TRIALS:-60}"
echo "✅ OPTUNA_TIMEOUT: ${OPTUNA_TIMEOUT:-None}"

# Créer le dossier models/ s'il n'existe pas
mkdir -p models

# Exécuter l'entraînement
echo "📊 Lancement de l'entraînement..."
python train_model.py

echo "✅ Entraînement terminé avec succès"
echo "Date de fin: $(date)"
```

Puis dans Render, modifiez le **Start Command** pour un service dédié :
```bash
bash scripts/train_on_render.sh
```

---

## 🎯 Recommandation finale

Pour un environnement de **production**, je recommande cette approche progressive :

### Phase 1 : Développement/Test
- Utiliser le **Shell Render** pour tester et déboguer
- Vérifier que tout fonctionne correctement

### Phase 2 : Automatisation basique
- Créer l'endpoint **`/admin/train-model`** avec authentification
- Tester manuellement via curl ou Postman

### Phase 3 : Automatisation complète
- Utiliser **GitHub Actions** pour déclencher l'entraînement
- Planifier l'entraînement hebdomadaire automatiquement

### Phase 4 : Production robuste (optionnel)
- Créer un **service séparé** (Background Worker) si l'entraînement est très long
- Utiliser une queue (Celery, RQ) pour gérer les tâches longues

---

## 📚 Ressources supplémentaires

- [Documentation Render Shell](https://render.com/docs/web-services#shell)
- [Documentation Render API](https://render.com/docs/api)
- [Documentation GitHub Actions](https://docs.github.com/en/actions)
- [Documentation Render Background Workers](https://render.com/docs/background-workers)

---

## 🔐 Sécurité

⚠️ **IMPORTANT** : L'endpoint `/admin/train-model` doit être sécurisé en production !

**Recommandations** :
1. Utiliser un token d'authentification fort
2. Limiter l'accès par IP (si possible)
3. Logger toutes les tentatives d'accès
4. Utiliser HTTPS uniquement
5. Ne pas exposer le token dans les logs

---

## ✅ Checklist avant d'exécuter train_model.py sur Render

- [ ] DATABASE_URL configurée dans Render
- [ ] Base de données PostgreSQL accessible depuis Render
- [ ] Variables d'environnement définies (OPTUNA_TRIALS, etc.)
- [ ] Fichier train_model.py présent dans le déploiement
- [ ] Dossier models/ créable/accessible
- [ ] Permissions correctes sur les fichiers
- [ ] Endpoint /admin/train-model créé (si méthode 2)
- [ ] Token ADMIN_TOKEN configuré (si méthode 2)
- [ ] Secrets GitHub configurés (si méthode 3)

---

**Bon entraînement ! 🚀**

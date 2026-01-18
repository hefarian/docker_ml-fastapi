# ============================================================================
# FICHIER : Dockerfile
# ============================================================================
#
# QU'EST-CE QU'UN DOCKERFILE ?
# ============================
# Un Dockerfile est un fichier texte qui contient toutes les instructions
# pour construire une image Docker. C'est comme une recette de cuisine :
# on dit à Docker comment installer les dépendances, copier les fichiers,
# et configurer l'environnement.
#
# COMMENT ÇA MARCHE ?
# ===================
# Chaque ligne (instruction) crée une "couche" dans l'image Docker.
# Docker met en cache ces couches, donc si vous modifiez juste le code,
# il n'a pas besoin de réinstaller toutes les dépendances.
#
# ORDRE DES INSTRUCTIONS :
# ========================
# 1. FROM : image de base (système d'exploitation + Python)
# 2. ENV : variables d'environnement
# 3. RUN : commandes à exécuter (installation de packages)
# 4. WORKDIR : répertoire de travail
# 5. COPY : copier des fichiers depuis votre PC vers l'image
# 6. EXPOSE : port à exposer
# 7. CMD : commande à exécuter au démarrage du conteneur
#
# ============================================================================

# ----------------------------------------------------------------------------
# ÉTAPE 1 : Image de base
# ----------------------------------------------------------------------------
# FROM : définit l'image de base à utiliser
# "python:3.11-slim" : image officielle Python 3.11
# "slim" = version allégée (plus petite, plus rapide à télécharger)
# Cette image contient déjà Python 3.11 et pip installés
FROM python:3.11-slim

# ----------------------------------------------------------------------------
# ÉTAPE 2 : Variables d'environnement Python
# ----------------------------------------------------------------------------
# ENV : définit des variables d'environnement dans l'image
# Ces variables sont disponibles dans tous les processus Python

# PYTHONDONTWRITEBYTECODE=1 : empêche Python de créer des fichiers .pyc
# Les .pyc sont des fichiers compilés Python (cache)
# En production, on n'en a pas besoin et ça prend de la place
ENV PYTHONDONTWRITEBYTECODE=1

# PYTHONUNBUFFERED=1 : désactive le buffer de sortie Python
# Les print() et logs s'affichent immédiatement (pas d'attente)
# Important pour voir les logs en temps réel dans Docker
ENV PYTHONUNBUFFERED=1

# ----------------------------------------------------------------------------
# ÉTAPE 3 : Installation des dépendances système
# ----------------------------------------------------------------------------
# RUN : exécute une commande dans l'image (comme dans un terminal)
# "apt-get" : gestionnaire de packages Linux (Debian/Ubuntu)

# Mise à jour de la liste des packages disponibles
# && : exécute la commande suivante seulement si la précédente réussit
# --no-install-recommends : n'installe que l'essentiel (réduit la taille)
# && rm -rf : supprime les fichiers temporaires pour réduire la taille de l'image
RUN apt-get update && apt-get install -y --no-install-recommends \
    # gcc : compilateur C (nécessaire pour compiler certains packages Python)
    gcc \
    # libpq-dev : bibliothèque PostgreSQL (nécessaire pour psycopg2)
    libpq-dev \
    # Supprime les fichiers temporaires pour réduire la taille de l'image
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------------
# ÉTAPE 4 : Définir le répertoire de travail
# ----------------------------------------------------------------------------
# WORKDIR : définit le répertoire courant pour toutes les commandes suivantes
# Si le répertoire n'existe pas, Docker le crée automatiquement
# Toutes les commandes suivantes (COPY, RUN, CMD) s'exécutent dans ce répertoire
WORKDIR /app

# ----------------------------------------------------------------------------
# ÉTAPE 5 : Installation des dépendances Python
# ----------------------------------------------------------------------------
# COPY : copie un fichier/dossier depuis votre PC vers l'image Docker
# "app/requirements.txt" : chemin sur votre PC (relatif au Dockerfile)
# "." : destination dans l'image (répertoire courant = /app)

# Copier le fichier requirements.txt AVANT de copier le code
# Pourquoi ? Docker met en cache les couches. Si le code change mais pas
# les dépendances, Docker réutilise la couche avec les packages installés
COPY app/requirements.txt .

# RUN : installer les packages Python
# --upgrade pip : met à jour pip vers la dernière version
# --no-cache-dir : ne garde pas les fichiers de cache pip (réduit la taille)
# -r requirements.txt : installe tous les packages listés dans requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# ----------------------------------------------------------------------------
# ÉTAPE 6 : Copier le code de l'application
# ----------------------------------------------------------------------------
# COPY : copie tout le contenu du dossier app/ vers /app dans l'image
# Cela inclut : main.py, model.py, schemas.py, etc.
COPY app .

# ----------------------------------------------------------------------------
# ÉTAPE 7 : Créer le dossier pour les modèles
# ----------------------------------------------------------------------------
# RUN mkdir -p : crée le dossier models/ même s'il est vide
# -p : crée aussi les dossiers parents si nécessaire
# Ce dossier sera utilisé pour stocker les modèles entraînés
RUN mkdir -p models

# ----------------------------------------------------------------------------
# ÉTAPE 8 : Configurer le PYTHONPATH
# ----------------------------------------------------------------------------
# ENV PYTHONPATH : indique à Python où chercher les modules
# /app : répertoire où se trouve le code
# Permet d'importer les modules avec "from schemas import ..." au lieu de chemins relatifs
ENV PYTHONPATH=/app

# ----------------------------------------------------------------------------
# ÉTAPE 9 : Exposer le port
# ----------------------------------------------------------------------------
# EXPOSE : indique que le conteneur écoute sur le port 8090
# Note : cela ne mappe PAS le port automatiquement
# Le mapping se fait dans docker-compose.yml avec "ports: - '8090:8090'"
EXPOSE 8090

# ----------------------------------------------------------------------------
# ÉTAPE 10 : Commande de démarrage
# ----------------------------------------------------------------------------
# CMD : commande exécutée au démarrage du conteneur
# Format : ["commande", "arg1", "arg2", ...]
# uvicorn : serveur ASGI pour FastAPI (équivalent de Apache/Nginx pour Python)
# main:app : module "main" (fichier main.py), variable "app" (instance FastAPI)
# --host 0.0.0.0 : écoute sur toutes les interfaces réseau (pas seulement localhost)
# --port 8090 : port d'écoute
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8090"]

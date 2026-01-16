# ---- Étape 1 : image de base ----
FROM python:3.11-slim

# ---- Étape 2 : variables d'environnement ----
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# ---- Étape 3 : installer les dépendances système ----
RUN apt-get update \
    && apt-get install -y gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- Étape 4 : mettre à jour pip ----
RUN pip install --upgrade pip

# ---- Étape 5 : copier les requirements ----
WORKDIR /app
COPY app/requirements.txt .

# ---- Étape 6 : installer les packages Python ----
RUN pip install --no-cache-dir -r requirements.txt

# ---- Étape 7 : copier le code de l'application ----
COPY app .

# ---- Étape 8 : exposer le port et démarrer Uvicorn ----
EXPOSE 8090
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8090"]
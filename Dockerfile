
# ---- Base ----
FROM python:3.11-slim

# ---- Env ----
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---- Deps système ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- Workdir ----
WORKDIR /app

# ---- Requirements ----
COPY app/requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# ---- Code ----
COPY app .

# ---- Models (créer le dossier même s'il est vide) ----
RUN mkdir -p models

# ---- PYTHONPATH pour les imports ----
ENV PYTHONPATH=/app

# ---- Expose & Run ----
EXPOSE 8090
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8090"]

# db/db.py
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import os
from functools import lru_cache

@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Crée une connexion SQLAlchemy à PostgreSQL"""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL non défini")
    engine = create_engine(database_url, pool_pre_ping=True)
    return engine

def ping_db():
    """Teste la connexion à la base de données"""
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))


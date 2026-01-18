"""
Module de gestion de la base de données PostgreSQL
"""
import os
import json
from typing import Optional, Dict, Any
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from psycopg2 import pool
from functools import lru_cache

class Database:
    """Gestionnaire de base de données PostgreSQL"""
    
    def __init__(self, database_url: str):
        """Initialise le pool de connexions"""
        self.pool = pool.SimpleConnectionPool(
            minconn=1,
            maxconn=5,
            dsn=database_url
        )
    
    def get_connection(self):
        """Obtient une connexion du pool"""
        return self.pool.getconn()
    
    def return_connection(self, conn):
        """Retourne une connexion au pool"""
        self.pool.putconn(conn)
    
    def save_prediction(
        self,
        input_data: Dict[str, Any],
        prediction: int,
        probability: float,
        model_version: str
    ) -> Optional[int]:
        """
        Enregistre une prédiction dans la base de données
        
        Returns:
            ID de la prédiction enregistrée
        """
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO predictions (input_data, prediction, probability, model_version)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        json.dumps(input_data),
                        prediction,
                        probability,
                        model_version
                    )
                )
                prediction_id = cur.fetchone()[0]
                conn.commit()
                return prediction_id
        except Exception as e:
            if conn:
                conn.rollback()
            raise Exception(f"Erreur lors de l'enregistrement de la prédiction: {e}")
        finally:
            if conn:
                self.return_connection(conn)
    
    def get_predictions(self, limit: int = 100) -> list:
        """Récupère les dernières prédictions"""
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, input_data, prediction, probability, model_version, created_at
                    FROM predictions
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (limit,)
                )
                return cur.fetchall()
        except Exception as e:
            raise Exception(f"Erreur lors de la récupération des prédictions: {e}")
        finally:
            if conn:
                self.return_connection(conn)
    
    def test_connection(self) -> bool:
        """Teste la connexion à la base de données"""
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                return True
        except Exception:
            return False
        finally:
            if conn:
                self.return_connection(conn)

@lru_cache(maxsize=1)
def get_database() -> Database:
    """Obtient une instance de la base de données (singleton)"""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL non défini")
    return Database(database_url)


"""
Module de gestion de la base de données PostgreSQL.

Ce module encapsule :
- La création d'un pool de connexions PostgreSQL (réutilisation des connexions).
- Des opérations de lecture/écriture sur une table `predictions`.
- Un helper "singleton" pour exposer une instance unique de Database.

Points clés :
- Utilise psycopg2 et SimpleConnectionPool (suffisant pour de nombreux services).
- Les données d'entrée (`input_data`) sont stockées en JSON (idéalement JSONB côté PostgreSQL).
- Gestion basique des transactions : commit sur succès, rollback sur erreur.
"""

import os
import json  # utilisé pour sérialiser `input_data` ; voir note plus bas pour JSONB natif
from typing import Optional, Dict, Any
# from datetime import datetime  # <-- import non utilisé dans ce module ; peut être supprimé
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values  # execute_values importé (peut servir pour du bulk)
from psycopg2 import pool
from functools import lru_cache


class Database:
    """Gestionnaire de base de données PostgreSQL.

    Cette classe :
    - Initialise un pool de connexions (1 à 5 connexions ici).
    - Fournit des méthodes pour :
        * save_prediction : insérer une prédiction et retourner son id.
        * get_predictions : lister les dernières prédictions.
        * test_connection : tester la disponibilité de la base.
    - Expose des helpers get_connection/return_connection pour gérer le cycle de vie des connexions.
    
    Remarque :
    - En production, il est souvent pratique d'ajouter une méthode `close()` pour fermer le pool
      proprement à l'arrêt du service (self.pool.closeall()).
    """

    def __init__(self, database_url: str):
        """Initialise le pool de connexions.

        Paramètres :
            database_url (str): DSN PostgreSQL (ex: 'postgresql://user:pwd@host:5432/dbname').

        Choix techniques :
        - SimpleConnectionPool : gère un nombre min et max de connexions. Ici min=1, max=5.
          Ajuster en fonction de la charge, du nombre de workers, et de `max_connections` côté PostgreSQL.
        """
        self.pool = pool.SimpleConnectionPool(
            minconn=1,          # nombre minimal de connexions conservées ouvertes
            maxconn=5,          # nombre maximal de connexions simultanées dans le pool
            dsn=database_url    # chaîne de connexion PostgreSQL
        )
    
    def get_connection(self):
        """Obtient une connexion disponible depuis le pool.

        Retour :
            connexion psycopg2 (objet Connection)

        Important :
        - Penser à toujours "rendre" la connexion via `return_connection` (finally),
          sinon fuite de connexions (le pool se vide).
        - Une alternative plus sûre est de fournir un context manager (with) pour garantir la restitution.
        """
        return self.pool.getconn()
    
    def return_connection(self, conn):
        """Retourne une connexion au pool après utilisation.

        Paramètres :
            conn : connexion psycopg2 précédemment extraite via get_connection

        Rôle :
        - Remet la connexion dans le pool pour être réutilisée par d'autres opérations.
        """
        self.pool.putconn(conn)
    
    def save_prediction(
        self,
        input_data: Dict[str, Any],
        prediction: int,
        probability: float,
        model_version: str
    ) -> Optional[int]:  # Remarque : en pratique, cette méthode renvoie toujours un int ou lève une exception.
        """
        Enregistre une prédiction dans la base de données.

        Paramètres :
            input_data (dict[str, Any]) : données d'entrée du modèle (sérialisées en JSON pour stockage)
            prediction (int)            : classe ou étiquette prédite
            probability (float)         : probabilité associée (idéalement contrainte CHECK 0..1 en DB)
            model_version (str)         : version du modèle (traçabilité MLOps)

        Retour :
            Optional[int] : ID de la prédiction enregistrée (en réalité, int si succès, lève sinon)

        Détails d'implémentation :
        - Utilise un curseur context manager (with conn.cursor()) pour s'assurer de la fermeture du curseur.
        - Sérialise input_data en JSON via json.dumps(). Côté PostgreSQL, la colonne devrait être de type JSONB.
          Alternative recommandée : psycopg2.extras.Json(input_data) pour déléguer la sérialisation.
        - `RETURNING id` permet de récupérer l'identifiant généré par la base en un seul round-trip.
        - commit() sur succès ; rollback() sur exception.
        - En cas d'erreur, on relance une Exception générique (perte du type d’erreur initial).
          En production, on préférera lever une exception dédiée (e.g. DatabaseError) avec `raise ... from e`.
        """
        conn = None
        try:
            conn = self.get_connection()  # extrait une connexion du pool
            with conn.cursor() as cur:
                # Paramétrage via placeholders %s pour prévenir l'injection SQL
                cur.execute(
                    """
                    INSERT INTO predictions (input_data, prediction, probability, model_version)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        json.dumps(input_data),  # sérialisation en JSON (voir alternative Json(...) ci-dessus)
                        prediction,
                        probability,
                        model_version
                    )
                )
                # fetchone() retourne un tuple (id,) -> on prend l'élément 0
                prediction_id = cur.fetchone()[0]
                conn.commit()  # valide la transaction
                return prediction_id
        except Exception as e:
            # En cas d'erreur, on annule la transaction pour éviter une session "salie"
            if conn:
                conn.rollback()
            # On relance une exception avec message contextualisé.
            # Suggestion : `raise DatabaseError(...) from e` pour garder la stack et le type d’erreur d’origine.
            raise Exception(f"Erreur lors de l'enregistrement de la prédiction: {e}")
        finally:
            # Quel que soit le résultat (succès/échec), on rend la connexion au pool si on l'a obtenue
            if conn:
                self.return_connection(conn)
    
    def get_predictions(self, limit: int = 100) -> list:
        """Récupère les dernières prédictions insérées.

        Paramètres :
            limit (int) : nombre maximal de lignes à renvoyer (défaut : 100)

        Retour :
            list : liste de lignes (dicts) grâce à RealDictCursor
                   Chaque élément contient les colonnes : id, input_data, prediction, probability,
                   model_version, created_at

        Détails :
        - ORDER BY created_at DESC permet d'obtenir les plus récentes d'abord.
        - RealDictCursor renvoie chaque ligne comme un dictionnaire (clé=nom de colonne) :
          très pratique pour exposer des APIs JSON.
        - En cas de volumétrie importante, prévoir des index (ex: sur created_at) et éventuellement
          une pagination de type "keyset" (WHERE created_at < ...) plutôt que OFFSET pour de meilleures perfs.
        """
        conn = None
        try:
            conn = self.get_connection()
            # cursor_factory=RealDictCursor -> fetchall() renvoie list[Dict[str, Any]]
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, input_data, prediction, probability, model_version, created_at
                    FROM predictions
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (limit,)  # tuple à un seul élément (note la virgule)
                )
                return cur.fetchall()
        except Exception as e:
            # Même remarque que dans save_prediction : on lève une Exception générique.
            # Pour une meilleure observabilité, définir une exception dédiée et logger l'erreur.
            raise Exception(f"Erreur lors de la récupération des prédictions: {e}")
        finally:
            if conn:
                self.return_connection(conn)
    
    def test_connection(self) -> bool:
        """Teste la connexion à la base de données.

        Principe :
        - Exécute une requête triviale 'SELECT 1'.
        - Retourne True si ça passe, False sinon.
        
        Utilité :
        - Idéal pour un endpoint /health ou /ready.
        """
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                return True
        except Exception:
            # En cas d'exception, on considère la base non joignable.
            return False
        finally:
            if conn:
                self.return_connection(conn)


@lru_cache(maxsize=1)
def get_database() -> Database:
    """Obtient une instance de la base de données (singleton).

    Comportement :
    - Lit la variable d'environnement DATABASE_URL.
    - Construit une instance Database et la met en cache (LRU) pour réutilisation.
      -> Lazy init : la première fois qu'on appelle get_database(), on crée l'instance.
      -> Appels suivants : on réutilise la même instance (et donc le même pool).

    Remarques :
    - Si DATABASE_URL change dynamiquement (rare), le cache conservera l'ancienne instance.
    - En environnement multi-process (ex: gunicorn), chaque process aura son propre pool (comportement souhaitable).
    - En complément, prévoir un hook d'arrêt pour fermer le pool (Database.close()) si nécessaire.
    """
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        # RuntimeError explicite si la config est absente (plutôt que fallback hasardeux).
        raise RuntimeError("DATABASE_URL non défini")
    return Database(database_url)


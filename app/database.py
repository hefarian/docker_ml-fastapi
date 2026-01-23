# ============================================================================
# FICHIER : app/database.py
# ============================================================================
#
# QU'EST-CE QUE CE FICHIER ?
# ===========================
# Ce fichier gère toutes les interactions avec la base de données PostgreSQL.
# Il encapsule :
# - La création d'un pool de connexions (réutilisation des connexions)
# - Les opérations de lecture/écriture sur la table `predictions`
# - Un helper "singleton" pour exposer une instance unique de Database
#
# QU'EST-CE QU'UN POOL DE CONNEXIONS ?
# ====================================
# Un pool de connexions est un ensemble de connexions à la base de données
# qui sont réutilisées au lieu d'être créées/détruites à chaque requête.
# Avantages :
# - Plus rapide (pas besoin de créer une connexion à chaque fois)
# - Plus efficace (limite le nombre de connexions simultanées)
# - Plus stable (gère les connexions qui se ferment)
#
# ============================================================================

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

import json  # utilisé pour sérialiser `input_data` ; voir note plus bas pour JSONB natif
import os
from functools import lru_cache
from typing import Any, Dict, Optional

# from datetime import datetime  # <-- import non utilisé dans ce module ; peut être supprimé
import psycopg2
from psycopg2 import pool
from psycopg2.extras import (  # execute_values importé (peut servir pour du bulk)
    RealDictCursor,
    execute_values,
)


# ============================================================================
# CLASSE : Database
# ============================================================================
class Database:
    """
    Gestionnaire de base de données PostgreSQL.

    QU'EST-CE QUE CETTE CLASSE ?
    ============================
    Cette classe gère toutes les interactions avec PostgreSQL :
    - Pool de connexions (réutilisation des connexions)
    - Sauvegarde des prédictions
    - Récupération des prédictions
    - Test de connexion

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
        """
        Initialise le pool de connexions.

        QU'EST-CE QU'UN POOL DE CONNEXIONS ?
        ====================================
        Un pool de connexions maintient un ensemble de connexions à la base de données
        qui sont réutilisées au lieu d'être créées/détruites à chaque requête.

        AVANTAGES :
        ===========
        - Performance : créer une connexion est lent, réutiliser est rapide
        - Efficacité : limite le nombre de connexions simultanées
        - Stabilité : gère automatiquement les connexions qui se ferment

        Paramètres :
            database_url (str): DSN PostgreSQL (ex: 'postgresql://user:pwd@host:5432/dbname').

        Choix techniques :
        - SimpleConnectionPool : gère un nombre min et max de connexions. Ici min=1, max=5.
          Ajuster en fonction de la charge, du nombre de workers, et de `max_connections` côté PostgreSQL.

        EXEMPLE D'URL :
        ===============
        postgresql://utilisateur:mot_de_passe@localhost:5432/nom_base
        """
        # Créer le pool de connexions
        # SimpleConnectionPool : pool simple mais efficace pour la plupart des cas
        self.pool = pool.SimpleConnectionPool(
            minconn=1,  # nombre minimal de connexions conservées ouvertes
            # Même si aucune requête n'est en cours, garder au moins 1 connexion ouverte
            maxconn=5,  # nombre maximal de connexions simultanées dans le pool
            # Si 5 connexions sont utilisées, les nouvelles requêtes attendent
            dsn=database_url,  # chaîne de connexion PostgreSQL
        )

    def get_connection(self):
        """
        Obtient une connexion disponible depuis le pool.

        QU'EST-CE QUE CETTE MÉTHODE FAIT ?
        ===================================
        Cette méthode extrait une connexion du pool pour l'utiliser.
        Si toutes les connexions sont utilisées, elle attend qu'une se libère.

        Retour :
            connexion psycopg2 (objet Connection)

        Important :
        - Penser à toujours "rendre" la connexion via `return_connection` (finally),
          sinon fuite de connexions (le pool se vide).
        - Une alternative plus sûre est de fournir un context manager (with) pour garantir la restitution.

        EXEMPLE D'UTILISATION :
        ======================
        conn = db.get_connection()
        try:
            # Utiliser la connexion
            ...
        finally:
            db.return_connection(conn)  # IMPORTANT : toujours rendre la connexion
        """
        return self.pool.getconn()

    def return_connection(self, conn):
        """
        Retourne une connexion au pool après utilisation.

        QU'EST-CE QUE CETTE MÉTHODE FAIT ?
        ===================================
        Cette méthode remet la connexion dans le pool pour qu'elle puisse être réutilisée.
        Si on oublie d'appeler cette méthode, le pool se vide et on ne peut plus faire de requêtes.

        Paramètres :
            conn : connexion psycopg2 précédemment extraite via get_connection

        Rôle :
        - Remet la connexion dans le pool pour être réutilisée par d'autres opérations.

        IMPORTANT :
        ===========
        Toujours appeler cette méthode après avoir utilisé une connexion !
        Utiliser un bloc try/finally pour garantir l'appel même en cas d'erreur.
        """
        self.pool.putconn(conn)

    def save_prediction(
        self, input_data: Dict[str, Any], prediction: int, probability: float, model_version: str
    ) -> Optional[int]:  # Remarque : en pratique, cette méthode renvoie toujours un int ou lève une exception.
        """
        Enregistre une prédiction dans la base de données.

        QU'EST-CE QUE CETTE MÉTHODE FAIT ?
        ===================================
        Cette méthode insère une nouvelle ligne dans la table `predictions` avec :
        - Les données d'entrée (input_data) : ce qui a été envoyé au modèle
        - La prédiction : le résultat du modèle (0 ou 1)
        - La probabilité : la confiance du modèle (0.0 à 1.0)
        - La version du modèle : pour savoir quel modèle a fait la prédiction

        Paramètres :
            input_data (dict[str, Any]) : données d'entrée du modèle (sérialisées en JSON pour stockage)
            prediction (int)            : classe ou étiquette prédite (0 = reste, 1 = quitte)
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
        - En cas d'erreur, on relance une Exception générique (perte du type d'erreur initial).
          En production, on préférera lever une exception dédiée (e.g. DatabaseError) avec `raise ... from e`.

        EXEMPLE D'UTILISATION :
        =======================
        >>> db = Database("postgresql://...")
        >>> prediction_id = db.save_prediction(
        ...     input_data={"age": 35, "genre": "M", ...},
        ...     prediction=1,
        ...     probability=0.75,
        ...     model_version="1.0.0"
        ... )
        >>> print(f"Prédiction enregistrée avec l'ID: {prediction_id}")
        """
        conn = None
        try:
            # ================================================================
            # ÉTAPE 1 : OBTENIR UNE CONNEXION
            # ================================================================
            conn = self.get_connection()  # extrait une connexion du pool

            # ================================================================
            # ÉTAPE 2 : EXÉCUTER LA REQUÊTE INSERT
            # ================================================================
            # with conn.cursor() : garantit que le curseur est fermé après utilisation
            with conn.cursor() as cur:
                # Paramétrage via placeholders %s pour prévenir l'injection SQL
                # IMPORTANT : ne jamais concaténer des strings dans les requêtes SQL !
                # Utiliser TOUJOURS des placeholders %s pour éviter les injections SQL
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
                        model_version,
                    ),
                )
                # ============================================================
                # ÉTAPE 3 : RÉCUPÉRER L'ID GÉNÉRÉ
                # ============================================================
                # fetchone() retourne un tuple (id,) -> on prend l'élément 0
                # RETURNING id : PostgreSQL retourne l'ID généré automatiquement
                prediction_id = cur.fetchone()[0]

                # ============================================================
                # ÉTAPE 4 : VALIDER LA TRANSACTION
                # ============================================================
                conn.commit()  # valide la transaction (sauvegarde les changements)
                return prediction_id
        except Exception as e:
            # ================================================================
            # GESTION DES ERREURS
            # ================================================================
            # En cas d'erreur, on annule la transaction pour éviter une session "salie"
            if conn:
                conn.rollback()  # Annule tous les changements de la transaction
            # On relance une exception avec message contextualisé.
            # Suggestion : `raise DatabaseError(...) from e` pour garder la stack et le type d'erreur d'origine.
            raise Exception(f"Erreur lors de l'enregistrement de la prédiction: {e}")
        finally:
            # ================================================================
            # NETTOYAGE : RENDRE LA CONNEXION AU POOL
            # ================================================================
            # Quel que soit le résultat (succès/échec), on rend la connexion au pool si on l'a obtenue
            # Le bloc finally garantit que cette ligne est TOUJOURS exécutée
            if conn:
                self.return_connection(conn)

    def get_predictions(self, limit: int = 100) -> list:
        """
        Récupère les dernières prédictions insérées.

        QU'EST-CE QUE CETTE MÉTHODE FAIT ?
        ===================================
        Cette méthode lit les dernières prédictions depuis la table `predictions`
        et les retourne dans l'ordre décroissant (plus récentes en premier).

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

        EXEMPLE D'UTILISATION :
        =======================
        >>> db = Database("postgresql://...")
        >>> predictions = db.get_predictions(limit=10)
        >>> for pred in predictions:
        ...     print(f"ID: {pred['id']}, Prédiction: {pred['prediction']}, Probabilité: {pred['probability']}")
        """
        conn = None
        try:
            # ================================================================
            # ÉTAPE 1 : OBTENIR UNE CONNEXION
            # ================================================================
            conn = self.get_connection()

            # ================================================================
            # ÉTAPE 2 : EXÉCUTER LA REQUÊTE SELECT
            # ================================================================
            # cursor_factory=RealDictCursor -> fetchall() renvoie list[Dict[str, Any]]
            # RealDictCursor : chaque ligne est un dictionnaire (plus pratique que des tuples)
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, input_data, prediction, probability, model_version, created_at
                    FROM predictions
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (limit,),  # tuple à un seul élément (note la virgule)
                )
                # ============================================================
                # ÉTAPE 3 : RÉCUPÉRER TOUTES LES LIGNES
                # ============================================================
                return cur.fetchall()
        except Exception as e:
            # Même remarque que dans save_prediction : on lève une Exception générique.
            # Pour une meilleure observabilité, définir une exception dédiée et logger l'erreur.
            raise Exception(f"Erreur lors de la récupération des prédictions: {e}")
        finally:
            # ================================================================
            # NETTOYAGE : RENDRE LA CONNEXION AU POOL
            # ================================================================
            if conn:
                self.return_connection(conn)

    def test_connection(self) -> bool:
        """
        Teste la connexion à la base de données.

        QU'EST-CE QUE CETTE MÉTHODE FAIT ?
        ===================================
        Cette méthode exécute la requête la plus simple possible ("SELECT 1")
        pour vérifier que la connexion à PostgreSQL fonctionne.

        Principe :
        - Exécute une requête triviale 'SELECT 1'.
        - Retourne True si ça passe, False sinon.

        Utilité :
        - Idéal pour un endpoint /health ou /ready.
        - Permet de vérifier rapidement si PostgreSQL est accessible.

        RETOUR :
        ========
        bool : True si la connexion fonctionne, False sinon

        EXEMPLE D'UTILISATION :
        =======================
        >>> db = Database("postgresql://...")
        >>> if db.test_connection():
        ...     print("Base de données accessible !")
        ... else:
        ...     print("Erreur de connexion")
        """
        conn = None
        try:
            # ================================================================
            # ÉTAPE 1 : OBTENIR UNE CONNEXION
            # ================================================================
            conn = self.get_connection()

            # ================================================================
            # ÉTAPE 2 : EXÉCUTER UNE REQUÊTE SIMPLE
            # ================================================================
            with conn.cursor() as cur:
                # SELECT 1 : la requête la plus simple possible
                # Si cette requête fonctionne, la connexion est OK
                cur.execute("SELECT 1")
                return True
        except Exception:
            # ================================================================
            # GESTION DES ERREURS
            # ================================================================
            # En cas d'exception, on considère la base non joignable.
            # On retourne False au lieu de lever une exception
            return False
        finally:
            # ================================================================
            # NETTOYAGE : RENDRE LA CONNEXION AU POOL
            # ================================================================
            if conn:
                self.return_connection(conn)


# ============================================================================
# FONCTION : get_database (Factory avec cache)
# ============================================================================
@lru_cache(maxsize=1)
def get_database() -> Database:
    """
    Obtient une instance de la base de données (singleton).

    QU'EST-CE QUE CETTE FONCTION ?
    ===============================
    Cette fonction crée une instance Database et la met en cache.
    C'est un pattern "singleton" : une seule instance partagée dans toute l'application.

    QU'EST-CE QUE @lru_cache ?
    ==========================
    @lru_cache est un décorateur Python qui met en cache le résultat d'une fonction.
    - Première fois : la fonction s'exécute et crée l'instance Database
    - Fois suivantes : retourne directement l'instance mise en cache
    - maxsize=1 : garde seulement 1 résultat en cache

    POURQUOI C'EST IMPORTANT ?
    ===========================
    - Évite de créer plusieurs pools de connexions (inefficace)
    - Garantit qu'une seule instance Database existe (singleton)
    - Économise la mémoire et les ressources

    Comportement :
    - Lit la variable d'environnement DATABASE_URL.
    - Construit une instance Database et la met en cache (LRU) pour réutilisation.
      -> Lazy init : la première fois qu'on appelle get_database(), on crée l'instance.
      -> Appels suivants : on réutilise la même instance (et donc le même pool).

    Remarques :
    - Si DATABASE_URL change dynamiquement (rare), le cache conservera l'ancienne instance.
    - En environnement multi-process (ex: gunicorn), chaque process aura son propre pool (comportement souhaitable).
    - En complément, prévoir un hook d'arrêt pour fermer le pool (Database.close()) si nécessaire.

    RETOUR :
    ========
    Database : instance de la classe Database (singleton)

    EXCEPTIONS :
    ===========
    RuntimeError : si DATABASE_URL n'est pas défini

    EXEMPLE D'UTILISATION :
    =======================
    >>> db = get_database()
    >>> predictions = db.get_predictions(limit=10)
    """
    # ========================================================================
    # ÉTAPE 1 : RÉCUPÉRER L'URL DE LA BASE DE DONNÉES
    # ========================================================================
    # Récupérer la variable d'environnement DATABASE_URL
    database_url = os.getenv("DATABASE_URL")

    # ========================================================================
    # ÉTAPE 2 : VÉRIFIER QUE L'URL EXISTE
    # ========================================================================
    if not database_url:
        # RuntimeError explicite si la config est absente (plutôt que fallback hasardeux).
        raise RuntimeError("DATABASE_URL non défini")

    # ========================================================================
    # ÉTAPE 3 : CRÉER L'INSTANCE (MISE EN CACHE AUTOMATIQUEMENT)
    # ========================================================================
    # Créer une instance Database
    # @lru_cache met automatiquement cette instance en cache
    # Les appels suivants retourneront la même instance
    return Database(database_url)

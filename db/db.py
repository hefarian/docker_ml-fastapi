# db/db.py
"""
Module d'utilitaires pour la connexion à PostgreSQL avec SQLAlchemy.

QU'EST-CE QUE SQLALCHEMY ?
==========================
SQLAlchemy est une bibliothèque Python qui permet de :
1. Se connecter à une base de données (PostgreSQL, MySQL, SQLite, etc.)
2. Exécuter des requêtes SQL de manière sécurisée
3. Gérer les connexions automatiquement (pool de connexions)

POURQUOI UTILISER SQLALCHEMY ICI ?
==================================
Dans ce projet, on utilise principalement psycopg2 (dans database.py) pour les opérations
de base de données. Ce module db.py fournit une alternative avec SQLAlchemy,
utile pour certaines opérations plus complexes ou pour compatibilité.

COMMENT ÇA MARCHE ?
===================
- create_engine() : crée un "moteur" de connexion à la base
- pool_pre_ping=True : vérifie que les connexions sont valides avant de les utiliser
- get_engine() : fonction qui retourne toujours le même moteur (singleton)
"""

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import os
from functools import lru_cache

@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """
    Crée une connexion SQLAlchemy à PostgreSQL.
    
    QU'EST-CE QUE @lru_cache ?
    ==========================
    @lru_cache est un "décorateur" Python qui met en cache le résultat d'une fonction.
    - La première fois qu'on appelle get_engine(), la fonction s'exécute et crée le moteur
    - Les fois suivantes, Python retourne directement le moteur mis en cache
    - maxsize=1 : on garde seulement 1 résultat en cache (le dernier)
    
    POURQUOI C'EST UTILE ?
    ======================
    - Évite de créer plusieurs connexions à la base (économise les ressources)
    - Plus rapide : pas besoin de recréer la connexion à chaque appel
    - Pattern "singleton" : une seule instance partagée dans toute l'application
    
    RETOUR :
    ========
    Engine : objet SQLAlchemy qui représente la connexion à la base de données
    
    EXEMPLE D'UTILISATION :
    =======================
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM sirh LIMIT 10"))
        rows = result.fetchall()
    """
    # Récupérer l'URL de la base de données depuis les variables d'environnement
    # Format : "postgresql://utilisateur:mot_de_passe@hôte:port/nom_base"
    database_url = os.getenv("DATABASE_URL")
    
    # Vérifier que la variable d'environnement existe
    if not database_url:
        # Si elle n'existe pas, on lève une erreur explicite
        # C'est mieux que de continuer avec une valeur par défaut qui ne fonctionnerait pas
        raise RuntimeError("DATABASE_URL non défini")
    
    # Créer le moteur SQLAlchemy
    # - database_url : l'URL de connexion
    # - pool_pre_ping=True : vérifie que les connexions sont valides avant utilisation
    #   (utile si la connexion a été fermée par le serveur)
    engine = create_engine(database_url, pool_pre_ping=True)
    
    return engine

def ping_db():
    """
    Teste la connexion à la base de données.
    
    QU'EST-CE QUE "PING" ?
    ======================
    Un "ping" est un test simple pour vérifier qu'une connexion fonctionne.
    Ici, on exécute "SELECT 1" qui est la requête la plus simple possible.
    Si ça fonctionne, la base est accessible. Si ça échoue, il y a un problème.
    
    COMMENT ÇA MARCHE ?
    ===================
    1. On récupère le moteur de connexion (via get_engine())
    2. On ouvre une connexion (with engine.connect())
    3. On exécute "SELECT 1" (la requête la plus simple)
    4. Si ça fonctionne, la base est OK. Si ça échoue, une exception est levée.
    
    EXEMPLE D'UTILISATION :
    =======================
    try:
        ping_db()
        print("Base de données accessible !")
    except Exception as e:
        print(f"Erreur de connexion : {e}")
    """
    # Récupérer le moteur de connexion (singleton, mis en cache)
    engine = get_engine()
    
    # Ouvrir une connexion (le "with" garantit que la connexion sera fermée après)
    with engine.connect() as conn:
        # Exécuter la requête la plus simple possible
        # text("SELECT 1") : crée un objet SQLAlchemy pour la requête SQL
        # conn.execute() : exécute la requête
        # Si la connexion fonctionne, cette ligne ne lève pas d'exception
        # Si la connexion ne fonctionne pas, une exception est levée
        conn.execute(text("SELECT 1"))

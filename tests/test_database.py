"""
Tests unitaires pour le module database.py

QU'EST-CE QUE CE FICHIER ?
==========================
Ce fichier contient des tests pour la classe Database qui gère les interactions
avec PostgreSQL (sauvegarde et récupération des prédictions).

POURQUOI TESTER database.py ?
==============================
- Vérifier que les prédictions sont correctement sauvegardées
- Vérifier que les prédictions sont correctement récupérées
- Vérifier la gestion des erreurs (connexion, requêtes SQL)
- Vérifier le pool de connexions

EXÉCUTION :
===========
pytest tests/test_database.py -v
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Configuration du PYTHONPATH (identique à test_api.py)
root_dir = Path(__file__).parent.parent
app_dir = root_dir / "app"
sys.path.insert(0, str(app_dir))
os.chdir(app_dir)

from database import Database, get_database


class TestDatabase:
    """
    Tests pour la classe Database.
    
    STRATÉGIE DE TEST :
    ===================
    - Utiliser des mocks pour simuler PostgreSQL (évite d'avoir besoin d'une vraie DB)
    - Tester les cas de succès et d'échec
    - Vérifier que les connexions sont correctement gérées (get/return)
    """
    
    def test_database_initialization(self):
        """
        Test de l'initialisation de la classe Database.
        
        CE QUE CE TEST VÉRIFIE :
        ========================
        - La classe Database peut être instanciée avec une URL valide
        - Un pool de connexions est créé
        """
        # Mock du pool de connexions pour éviter de créer une vraie connexion
        with patch('database.pool.SimpleConnectionPool') as mock_pool:
            mock_pool_instance = MagicMock()
            mock_pool.return_value = mock_pool_instance
            
            # Créer une instance Database
            db = Database("postgresql://user:pass@localhost:5432/testdb")
            
            # Vérifier que le pool a été créé
            assert db.pool is not None
    
    def test_get_connection(self):
        """
        Test de la méthode get_connection.
        
        CE QUE CE TEST VÉRIFIE :
        ========================
        - get_connection() retourne une connexion depuis le pool
        """
        with patch('database.pool.SimpleConnectionPool') as mock_pool:
            mock_connection = MagicMock()
            mock_pool_instance = MagicMock()
            mock_pool_instance.getconn.return_value = mock_connection
            mock_pool.return_value = mock_pool_instance
            
            db = Database("postgresql://user:pass@localhost:5432/testdb")
            conn = db.get_connection()
            
            # Vérifier que getconn() a été appelé
            mock_pool_instance.getconn.assert_called_once()
            assert conn == mock_connection
    
    def test_return_connection(self):
        """
        Test de la méthode return_connection.
        
        CE QUE CE TEST VÉRIFIE :
        ========================
        - return_connection() remet la connexion dans le pool
        """
        with patch('database.pool.SimpleConnectionPool') as mock_pool:
            mock_connection = MagicMock()
            mock_pool_instance = MagicMock()
            mock_pool.return_value = mock_pool_instance
            
            db = Database("postgresql://user:pass@localhost:5432/testdb")
            db.return_connection(mock_connection)
            
            # Vérifier que putconn() a été appelé avec la bonne connexion
            mock_pool_instance.putconn.assert_called_once_with(mock_connection)
    
    def test_save_prediction_success(self):
        """
        Test de sauvegarde d'une prédiction (cas de succès).
        
        CE QUE CE TEST VÉRIFIE :
        ========================
        - save_prediction() insère correctement une prédiction
        - Retourne l'ID de la prédiction insérée
        - La transaction est commitée
        """
        with patch('database.pool.SimpleConnectionPool') as mock_pool:
            # Mock de la connexion et du curseur
            mock_connection = MagicMock()
            mock_cursor = MagicMock()
            mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
            mock_connection.cursor.return_value.__exit__.return_value = None
            
            # Simuler un ID retourné par RETURNING id
            mock_cursor.fetchone.return_value = (42,)  # ID = 42
            
            mock_pool_instance = MagicMock()
            mock_pool_instance.getconn.return_value = mock_connection
            mock_pool.return_value = mock_pool_instance
            
            db = Database("postgresql://user:pass@localhost:5432/testdb")
            
            # Données de test
            input_data = {"age": 35, "genre": "M"}
            prediction_id = db.save_prediction(
                input_data=input_data,
                prediction=1,
                probability=0.75,
                model_version="1.0.0"
            )
            
            # Vérifications
            assert prediction_id == 42
            mock_cursor.execute.assert_called_once()
            mock_connection.commit.assert_called_once()
            mock_connection.rollback.assert_not_called()
    
    def test_save_prediction_error_rollback(self):
        """
        Test de sauvegarde d'une prédiction (cas d'erreur).
        
        CE QUE CE TEST VÉRIFIE :
        ========================
        - En cas d'erreur, la transaction est rollback
        - Une exception est levée avec un message approprié
        """
        with patch('database.pool.SimpleConnectionPool') as mock_pool:
            # Mock de la connexion qui lève une exception
            mock_connection = MagicMock()
            mock_cursor = MagicMock()
            mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
            mock_connection.cursor.return_value.__exit__.return_value = None
            
            # Simuler une erreur lors de l'exécution
            mock_cursor.execute.side_effect = Exception("Erreur SQL")
            
            mock_pool_instance = MagicMock()
            mock_pool_instance.getconn.return_value = mock_connection
            mock_pool.return_value = mock_pool_instance
            
            db = Database("postgresql://user:pass@localhost:5432/testdb")
            
            # Tester que l'exception est levée
            with pytest.raises(Exception) as exc_info:
                db.save_prediction(
                    input_data={"age": 35},
                    prediction=1,
                    probability=0.75,
                    model_version="1.0.0"
                )
            
            # Vérifications
            assert "Erreur lors de l'enregistrement" in str(exc_info.value)
            mock_connection.rollback.assert_called_once()
            mock_connection.commit.assert_not_called()
    
    def test_get_predictions_success(self):
        """
        Test de récupération des prédictions (cas de succès).
        
        CE QUE CE TEST VÉRIFIE :
        ========================
        - get_predictions() retourne une liste de prédictions
        - Le curseur utilise RealDictCursor (retourne des dicts)
        - La limite est respectée
        """
        with patch('database.pool.SimpleConnectionPool') as mock_pool:
            # Mock de la connexion et du curseur
            mock_connection = MagicMock()
            mock_cursor = MagicMock()
            mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
            mock_connection.cursor.return_value.__exit__.return_value = None
            
            # Simuler des prédictions retournées
            mock_predictions = [
                {"id": 1, "prediction": 0, "probability": 0.3},
                {"id": 2, "prediction": 1, "probability": 0.8}
            ]
            mock_cursor.fetchall.return_value = mock_predictions
            
            mock_pool_instance = MagicMock()
            mock_pool_instance.getconn.return_value = mock_connection
            mock_pool.return_value = mock_pool_instance
            
            db = Database("postgresql://user:pass@localhost:5432/testdb")
            
            # Récupérer les prédictions
            predictions = db.get_predictions(limit=10)
            
            # Vérifications
            assert len(predictions) == 2
            assert predictions[0]["id"] == 1
            assert predictions[1]["id"] == 2
            mock_cursor.execute.assert_called_once()
    
    def test_get_predictions_error(self):
        """
        Test de récupération des prédictions (cas d'erreur).
        
        CE QUE CE TEST VÉRIFIE :
        ========================
        - En cas d'erreur, une exception est levée
        """
        with patch('database.pool.SimpleConnectionPool') as mock_pool:
            # Mock de la connexion qui lève une exception
            mock_connection = MagicMock()
            mock_cursor = MagicMock()
            mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
            mock_connection.cursor.return_value.__exit__.return_value = None
            
            # Simuler une erreur
            mock_cursor.execute.side_effect = Exception("Erreur SQL")
            
            mock_pool_instance = MagicMock()
            mock_pool_instance.getconn.return_value = mock_connection
            mock_pool.return_value = mock_pool_instance
            
            db = Database("postgresql://user:pass@localhost:5432/testdb")
            
            # Tester que l'exception est levée
            with pytest.raises(Exception) as exc_info:
                db.get_predictions(limit=10)
            
            assert "Erreur lors de la récupération" in str(exc_info.value)
    
    def test_test_connection_success(self):
        """
        Test de la méthode test_connection (cas de succès).
        
        CE QUE CE TEST VÉRIFIE :
        ========================
        - test_connection() retourne True si la connexion fonctionne
        """
        with patch('database.pool.SimpleConnectionPool') as mock_pool:
            # Mock de la connexion
            mock_connection = MagicMock()
            mock_cursor = MagicMock()
            mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
            mock_connection.cursor.return_value.__exit__.return_value = None
            
            mock_pool_instance = MagicMock()
            mock_pool_instance.getconn.return_value = mock_connection
            mock_pool.return_value = mock_pool_instance
            
            db = Database("postgresql://user:pass@localhost:5432/testdb")
            
            # Tester la connexion
            result = db.test_connection()
            
            # Vérifications
            assert result is True
            mock_cursor.execute.assert_called_once_with("SELECT 1")
    
    def test_test_connection_error(self):
        """
        Test de la méthode test_connection (cas d'erreur).
        
        CE QUE CE TEST VÉRIFIE :
        ========================
        - test_connection() retourne False en cas d'erreur
        """
        with patch('database.pool.SimpleConnectionPool') as mock_pool:
            # Mock de la connexion qui lève une exception
            mock_connection = MagicMock()
            mock_cursor = MagicMock()
            mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
            mock_connection.cursor.return_value.__exit__.return_value = None
            
            # Simuler une erreur
            mock_cursor.execute.side_effect = Exception("Erreur de connexion")
            
            mock_pool_instance = MagicMock()
            mock_pool_instance.getconn.return_value = mock_connection
            mock_pool.return_value = mock_pool_instance
            
            db = Database("postgresql://user:pass@localhost:5432/testdb")
            
            # Tester la connexion
            result = db.test_connection()
            
            # Vérifications
            assert result is False


class TestGetDatabase:
    """
    Tests pour la fonction get_database (singleton).
    
    STRATÉGIE DE TEST :
    ===================
    - Tester que get_database() crée une instance Database
    - Tester que le cache fonctionne (même instance retournée)
    - Tester l'erreur si DATABASE_URL n'est pas défini
    """
    
    def test_get_database_success(self):
        """
        Test de get_database() avec DATABASE_URL définie.
        
        CE QUE CE TEST VÉRIFIE :
        ========================
        - get_database() crée une instance Database si DATABASE_URL est définie
        """
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://user:pass@localhost:5432/testdb"}):
            with patch('database.Database') as mock_database:
                mock_instance = MagicMock()
                mock_database.return_value = mock_instance
                
                # Réinitialiser le cache (nécessaire pour les tests)
                get_database.cache_clear()
                
                # Appeler get_database()
                db = get_database()
                
                # Vérifier que Database a été instanciée
                mock_database.assert_called_once_with("postgresql://user:pass@localhost:5432/testdb")
                assert db == mock_instance
    
    def test_get_database_missing_url(self):
        """
        Test de get_database() sans DATABASE_URL.
        
        CE QUE CE TEST VÉRIFIE :
        ========================
        - get_database() lève une RuntimeError si DATABASE_URL n'est pas définie
        """
        # S'assurer que DATABASE_URL n'est pas définie
        if "DATABASE_URL" in os.environ:
            del os.environ["DATABASE_URL"]
        
        # Réinitialiser le cache
        get_database.cache_clear()
        
        # Tester que l'exception est levée
        with pytest.raises(RuntimeError) as exc_info:
            get_database()
        
        assert "DATABASE_URL non défini" in str(exc_info.value)
    
    def test_get_database_cache(self):
        """
        Test du cache de get_database().
        
        CE QUE CE TEST VÉRIFIE :
        ========================
        - get_database() retourne la même instance (cache)
        """
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://user:pass@localhost:5432/testdb"}):
            with patch('database.Database') as mock_database:
                mock_instance = MagicMock()
                mock_database.return_value = mock_instance
                
                # Réinitialiser le cache
                get_database.cache_clear()
                
                # Appeler get_database() deux fois
                db1 = get_database()
                db2 = get_database()
                
                # Vérifier que c'est la même instance (cache)
                assert db1 is db2
                # Database ne doit être instanciée qu'une seule fois
                assert mock_database.call_count == 1

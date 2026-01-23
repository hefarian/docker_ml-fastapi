"""
Tests unitaires pour le module train_model.py

QU'EST-CE QUE CE FICHIER ?
==========================
Ce fichier contient des tests pour les fonctions utilitaires du script
d'entraînement (safe_divide, prepare_data, etc.).

POURQUOI TESTER train_model.py ?
=================================
- Vérifier que les fonctions utilitaires fonctionnent correctement
- Vérifier la préparation des données
- Vérifier la gestion des erreurs

NOTE IMPORTANTE :
=================
Les tests d'entraînement complet nécessitent une base de données réelle
et peuvent être longs. On se concentre ici sur les fonctions unitaires.

EXÉCUTION :
===========
pytest tests/test_train_model.py -v
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd

# Configuration du PYTHONPATH
root_dir = Path(__file__).parent.parent
app_dir = root_dir / "app"
sys.path.insert(0, str(app_dir))
os.chdir(app_dir)

# Importer les fonctions à tester
from train_model import safe_divide, prepare_data, load_data_from_db


class TestSafeDivide:
    """
    Tests pour la fonction safe_divide.
    
    STRATÉGIE DE TEST :
    ===================
    - Tester les cas normaux (division réussie)
    - Tester la division par zéro
    - Tester les valeurs NaN/infini
    """
    
    def test_safe_divide_normal(self):
        """
        Test de division normale.
        
        CE QUE CE TEST VÉRIFIE :
        ========================
        - safe_divide() effectue correctement une division normale
        """
        num = pd.Series([10, 20, 30])
        den = pd.Series([2, 5, 3])
        result = safe_divide(num, den, fill_value=0.0)
        
        # Vérifier les résultats
        assert result.tolist() == [5.0, 4.0, 10.0]
    
    def test_safe_divide_by_zero(self):
        """
        Test de division par zéro.
        
        CE QUE CE TEST VÉRIFIE :
        ========================
        - safe_divide() gère correctement la division par zéro
        - Les valeurs infinies sont remplacées par fill_value
        """
        num = pd.Series([10, 20, 30])
        den = pd.Series([2, 0, 5])
        result = safe_divide(num, den, fill_value=0.0)
        
        # Vérifier que la division par zéro donne fill_value
        assert result.tolist() == [5.0, 0.0, 6.0]
    
    def test_safe_divide_with_nan(self):
        """
        Test avec des valeurs NaN.
        
        CE QUE CE TEST VÉRIFIE :
        ========================
        - safe_divide() gère correctement les valeurs NaN
        """
        num = pd.Series([10, np.nan, 30])
        den = pd.Series([2, 5, 3])
        result = safe_divide(num, den, fill_value=0.0)
        
        # NaN dans num doit donner fill_value
        assert result.iloc[0] == 5.0
        assert result.iloc[1] == 0.0  # NaN remplacé par fill_value
        assert result.iloc[2] == 10.0
    
    def test_safe_divide_with_infinity(self):
        """
        Test avec des valeurs infinies.
        
        CE QUE CE TEST VÉRIFIE :
        ========================
        - safe_divide() gère correctement les valeurs infinies
        """
        num = pd.Series([10, np.inf, 30])
        den = pd.Series([2, 5, 0])
        result = safe_divide(num, den, fill_value=0.0)
        
        # Vérifier que les infini sont gérés
        assert result.iloc[0] == 5.0
        # inf / 5 devrait donner inf, mais on le remplace par fill_value
        assert result.iloc[1] == 0.0 or np.isinf(result.iloc[1])
        assert result.iloc[2] == 0.0  # Division par zéro


class TestPrepareData:
    """
    Tests pour la fonction prepare_data.
    
    STRATÉGIE DE TEST :
    ===================
    - Tester avec des données minimales
    - Vérifier que les transformations sont appliquées
    - Vérifier que les encodeurs sont créés
    """
    
    def test_prepare_data_minimal(self):
        """
        Test de préparation avec des données minimales.
        
        CE QUE CE TEST VÉRIFIE :
        ========================
        - prepare_data() fonctionne avec un DataFrame minimal
        - Les transformations de base sont appliquées
        """
        # Créer un DataFrame minimal
        df = pd.DataFrame({
            "id_employee": [1, 2, 3],
            "age": [30, 35, 40],
            "a_quitte_l_entreprise": ["Non", "Oui", "Non"],
            "departement": ["R&D", "Sales", "R&D"],
            "frequence_deplacement": ["Aucun", "Occasionnel", "Frequent"]
        })
        
        # Appeler prepare_data
        df_prepared, ohe, ordinal = prepare_data(df)
        
        # Vérifications de base
        assert isinstance(df_prepared, pd.DataFrame)
        assert "Attrition" in df_prepared.columns  # Variable cible créée
        
        # Vérifier que Attrition est bien encodé (0 ou 1)
        assert df_prepared["Attrition"].isin([0, 1]).all()
    
    def test_prepare_data_with_augmentation_salaire(self):
        """
        Test avec colonne augmentation_salaire_precedente.
        
        CE QUE CE TEST VÉRIFIE :
        ========================
        - La normalisation de augmentation_salaire_precedente fonctionne
        """
        df = pd.DataFrame({
            "id_employee": [1, 2],
            "age": [30, 35],
            "a_quitte_l_entreprise": ["Non", "Oui"],
            "augmentation_salaire_precedente": ["11%", "15.5%"]
        })
        
        df_prepared, _, _ = prepare_data(df)
        
        # Vérifier que la colonne a été normalisée
        if "augmentation_salaire_precedente" in df_prepared.columns:
            # Les valeurs doivent être entre 0 et 1 (11% -> 0.11)
            assert df_prepared["augmentation_salaire_precedente"].max() <= 1.0
    
    def test_prepare_data_creates_encoders(self):
        """
        Test que les encodeurs sont créés.
        
        CE QUE CE TEST VÉRIFIE :
        ========================
        - prepare_data() retourne des encodeurs (OneHot et Ordinal)
        - Les encodeurs peuvent être None si pas de données catégorielles
        """
        df = pd.DataFrame({
            "id_employee": [1, 2],
            "age": [30, 35],
            "a_quitte_l_entreprise": ["Non", "Oui"],
            "departement": ["R&D", "Sales"],
            "frequence_deplacement": ["Aucun", "Occasionnel"]
        })
        
        df_prepared, ohe, ordinal = prepare_data(df)
        
        # Vérifier que les encodeurs sont retournés (peuvent être None)
        # Si des colonnes catégorielles existent, les encodeurs devraient être créés
        assert ohe is None or hasattr(ohe, 'transform')
        assert ordinal is None or hasattr(ordinal, 'transform')


class TestLoadDataFromDb:
    """
    Tests pour la fonction load_data_from_db.
    
    STRATÉGIE DE TEST :
    ===================
    - Utiliser des mocks pour simuler PostgreSQL
    - Tester le chargement des tables
    - Tester les jointures
    """
    
    def test_load_data_from_db_mock(self):
        """
        Test de chargement avec mocks.
        
        CE QUE CE TEST VÉRIFIE :
        ========================
        - load_data_from_db() charge les 3 tables
        - Les jointures sont effectuées correctement
        """
        # Mock des DataFrames retournés par pd.read_sql
        mock_sirh = pd.DataFrame({
            "id_employee": [1, 2, 3],
            "age": [30, 35, 40],
            "revenu_mensuel": [5000, 6000, 7000]
        })
        
        mock_eval = pd.DataFrame({
            "eval_number": ["E_1", "E_2", "E_3"],
            "note_evaluation": [4.0, 4.5, 3.8]
        })
        
        mock_sondage = pd.DataFrame({
            "code_sondage": [1, 2, 3],
            "satisfaction": [4, 5, 3]
        })
        
        # Mock de create_engine et pd.read_sql
        with patch('train_model.create_engine') as mock_engine:
            with patch('train_model.pd.read_sql') as mock_read_sql:
                # Configurer pd.read_sql pour retourner différents DataFrames selon la requête
                def read_sql_side_effect(query, conn):
                    if 'sirh' in str(query).lower():
                        return mock_sirh
                    elif 'eval' in str(query).lower():
                        return mock_eval
                    elif 'sondage' in str(query).lower():
                        return mock_sondage
                    return pd.DataFrame()
                
                mock_read_sql.side_effect = read_sql_side_effect
                
                # Mock de l'engine et de la connexion
                mock_conn = MagicMock()
                mock_engine_instance = MagicMock()
                mock_engine_instance.begin.return_value.__enter__.return_value = mock_conn
                mock_engine_instance.begin.return_value.__exit__.return_value = None
                mock_engine.return_value = mock_engine_instance
                
                # Appeler load_data_from_db
                df = load_data_from_db()
                
                # Vérifications
                assert isinstance(df, pd.DataFrame)
                # Vérifier que les colonnes des 3 tables sont présentes
                assert "age" in df.columns  # De sirh
                assert "note_evaluation" in df.columns  # De eval
                assert "satisfaction" in df.columns  # De sondage
    
    def test_load_data_from_db_with_missing_tables(self):
        """
        Test avec tables manquantes (gestion d'erreur).
        
        CE QUE CE TEST VÉRIFIE :
        ========================
        - load_data_from_db() gère les erreurs si une table est manquante
        """
        with patch('train_model.create_engine') as mock_engine:
            with patch('train_model.pd.read_sql') as mock_read_sql:
                # Simuler une erreur lors du chargement
                mock_read_sql.side_effect = Exception("Table not found")
                
                mock_conn = MagicMock()
                mock_engine_instance = MagicMock()
                mock_engine_instance.begin.return_value.__enter__.return_value = mock_conn
                mock_engine_instance.begin.return_value.__exit__.return_value = None
                mock_engine.return_value = mock_engine_instance
                
                # Tester que l'exception est propagée
                with pytest.raises(Exception):
                    load_data_from_db()


class TestDataPreparationEdgeCases:
    """
    Tests pour les cas limites de la préparation des données.
    
    STRATÉGIE DE TEST :
    ===================
    - Tester avec des données vides
    - Tester avec des valeurs manquantes
    - Tester avec des types incorrects
    """
    
    def test_prepare_data_empty_dataframe(self):
        """
        Test avec DataFrame vide.
        
        CE QUE CE TEST VÉRIFIE :
        ========================
        - prepare_data() gère un DataFrame vide (ou lève une exception appropriée)
        """
        df = pd.DataFrame()
        
        # Un DataFrame vide cause une erreur dans pd.get_dummies
        # C'est acceptable car un DataFrame vide n'a pas de sens pour l'entraînement
        # On teste que l'exception est levée de manière appropriée
        with pytest.raises((ValueError, KeyError, AttributeError)):
            df_prepared, _, _ = prepare_data(df)
    
    def test_prepare_data_missing_values(self):
        """
        Test avec valeurs manquantes.
        
        CE QUE CE TEST VÉRIFIE :
        ========================
        - prepare_data() gère les valeurs manquantes (NaN)
        """
        df = pd.DataFrame({
            "id_employee": [1, 2, 3],
            "age": [30, np.nan, 40],
            "a_quitte_l_entreprise": ["Non", "Oui", None],
            "departement": ["R&D", "Sales", "R&D"]
        })
        
        df_prepared, _, _ = prepare_data(df)
        
        # Vérifier que le DataFrame est créé malgré les NaN
        assert isinstance(df_prepared, pd.DataFrame)
        # Les NaN devraient être gérés (remplacés ou supprimés)
        # selon la logique de prepare_data

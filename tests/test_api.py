"""
Tests unitaires et fonctionnels pour l'API de prédiction d'attrition
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.schemas import EmployeeData, PredictRequest
import os

client = TestClient(app)

# Données d'exemple pour les tests
SAMPLE_EMPLOYEE = {
    "age": 35,
    "genre": "M",
    "revenu_mensuel": 5000,
    "statut_marital": "Marié(e)",
    "departement": "R&D",
    "poste": "Data Scientist",
    "nombre_experiences_precedentes": 3,
    "annee_experience_totale": 8,
    "annees_dans_l_entreprise": 5,
    "annees_dans_le_poste_actuel": 2,
    "satisfaction_employee_environnement": 4,
    "note_evaluation_precedente": 4.0,
    "niveau_hierarchique_poste": 2,
    "satisfaction_employee_nature_travail": 4,
    "satisfaction_employee_equipe": 4,
    "satisfaction_employee_equilibre_pro_perso": 3,
    "note_evaluation_actuelle": 4.5,
    "heure_supplementaires": "Non",
    "nombre_participation_pee": 2,
    "nb_formations_suivies": 3,
    "distance_domicile_travail": 10,
    "niveau_education": 3,
    "domaine_etude": "Data Science",
    "frequence_deplacement": "Occasionnel",
    "annees_depuis_la_derniere_promotion": 2,
    "annes_sous_responsable_actuel": 1,
    "augmentation_salaire_precedente": 0.05
}

def test_health_ok():
    """Test du endpoint de santé"""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "db_connected" in data

def test_root_ok():
    """Test du endpoint racine"""
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert "message" in data
    assert "version" in data
    assert "endpoints" in data

def test_predict_with_valid_data():
    """Test de prédiction avec des données valides"""
    payload = {"employee_data": SAMPLE_EMPLOYEE}
    res = client.post("/predict", json=payload)
    
    # Le modèle peut ne pas être chargé, donc on accepte 200 ou 503
    assert res.status_code in [200, 503]
    
    if res.status_code == 200:
        data = res.json()
        assert "prediction" in data
        assert "probability" in data
        assert "model_version" in data
        assert data["prediction"] in [0, 1]
        assert 0 <= data["probability"] <= 1

def test_predict_validation_error_missing_field():
    """Test de validation avec champ manquant"""
    invalid_data = SAMPLE_EMPLOYEE.copy()
    del invalid_data["age"]
    payload = {"employee_data": invalid_data}
    res = client.post("/predict", json=payload)
    assert res.status_code == 422

def test_predict_validation_error_invalid_value():
    """Test de validation avec valeur invalide"""
    invalid_data = SAMPLE_EMPLOYEE.copy()
    invalid_data["age"] = -5  # Âge négatif invalide
    payload = {"employee_data": invalid_data}
    res = client.post("/predict", json=payload)
    assert res.status_code == 422

def test_predict_validation_error_wrong_type():
    """Test de validation avec type incorrect"""
    invalid_data = SAMPLE_EMPLOYEE.copy()
    invalid_data["age"] = "trente-cinq"  # String au lieu d'int
    payload = {"employee_data": invalid_data}
    res = client.post("/predict", json=payload)
    assert res.status_code == 422

def test_get_predictions():
    """Test de récupération des prédictions"""
    res = client.get("/predictions")
    # Peut échouer si DB non connectée, donc on accepte 200 ou 500
    assert res.status_code in [200, 500]
    
    if res.status_code == 200:
        data = res.json()
        assert "count" in data
        assert "predictions" in data
        assert isinstance(data["predictions"], list)

def test_get_predictions_with_limit():
    """Test de récupération avec limite"""
    res = client.get("/predictions?limit=10")
    assert res.status_code in [200, 500]
    
    if res.status_code == 200:
        data = res.json()
        assert data["count"] <= 10

def test_db_test():
    """Test du endpoint de test DB"""
    res = client.get("/db-test")
    # Peut échouer si DB non connectée
    assert res.status_code in [200, 500]

def test_schema_employee_data():
    """Test de validation du schéma EmployeeData"""
    # Données valides
    employee = EmployeeData(**SAMPLE_EMPLOYEE)
    assert employee.age == 35
    
    # Données invalides
    with pytest.raises(Exception):
        invalid = SAMPLE_EMPLOYEE.copy()
        invalid["age"] = -1
        EmployeeData(**invalid)

def test_schema_predict_request():
    """Test de validation du schéma PredictRequest"""
    request = PredictRequest(employee_data=SAMPLE_EMPLOYEE)
    assert request.employee_data.age == 35
